"""Dynamic line rating (DLR) for overhead conductors.

Converts conductor parameters plus weather evidence (ambient temperature,
wind speed/direction, solar irradiance) and the present loading into an
*available capacity* (ampacity headroom) that carries an explicit validity
window.

Operating rules enforced here:

* Weather evidence is valid only while its timestamp is within
  ``valid_for_s`` of the rating time.  When evidence is missing, stale, or
  has bad quality the result falls back to the static nameplate rating
  (``basis == "static_fallback"``) and the fallback reason is recorded.
* Every result carries a ``reasons`` tuple explaining the outcome:
  fallback causes (``weather_missing``, ``weather_expired``,
  ``weather_invalid``), derating causes (``high_ambient``, ``low_wind``,
  ``solar_heating``) with the ``derated`` flag, uplift causes
  (``low_ambient``, ``favorable_wind``, ``night_cooling``) with the
  ``uplift`` flag, and ``overload`` when the present current exceeds the
  dynamic ampacity.
* Dynamic results are computed values.  They are returned as separate
  objects and are NEVER written back to the conductor's static nameplate
  ampacity -- temporary margin must not become equipment rating data.

The thermal model is a steady-state heat balance (IEEE 738 style):
joule heating + solar gain = forced/natural convection + radiation,
solved for the current that holds the conductor at its maximum
allowable temperature.  Standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from relaygrid.conductors.catalog import get as get_conductor

_STEFAN_BOLTZMANN = 5.6704e-8
_AIR_PRANDTL = 0.71
_RATING_BASIS_TEMP_C = 25.0  # typical static nameplate rating basis


@dataclass(frozen=True)
class LineConductor:
    """Physical and nameplate data for an overhead line conductor.

    ``static_ampacity_a`` is the nameplate rating and must never be
    overwritten with dynamically computed margin.
    """

    line_id: str
    r_ohm_per_m: float
    diameter_m: float
    static_ampacity_a: float
    max_temp_c: float = 75.0
    reference_temp_c: float = 25.0
    temp_coeff_per_c: float = 0.004
    emissivity: float = 0.9
    absorptivity: float = 0.8
    kv_ll: float | None = None


@dataclass(frozen=True)
class WeatherEvidence:
    """Weather observation with an evidence timestamp and validity window."""

    ambient_c: float
    wind_mps: float
    solar_w_m2: float
    timestamp_s: float
    valid_for_s: float = 1800.0
    wind_angle_deg: float = 90.0
    quality: str = "good"  # good | suspect | bad


@dataclass(frozen=True)
class DynamicLineRating:
    """Available capacity for one line at one point in time.

    ``basis`` is ``"dynamic"`` when fresh weather evidence was used and
    ``"static_fallback"`` when evidence was missing/expired/bad.
    """

    line_id: str
    basis: str
    static_ampacity_a: float
    dynamic_ampacity_a: float
    current_a: float
    available_a: float
    valid_from_s: float
    valid_until_s: float
    reasons: tuple[str, ...]
    conductor_temp_c: float | None = None
    ambient_c: float | None = None
    wind_mps: float | None = None
    solar_w_m2: float | None = None
    weather_age_s: float | None = None
    remaining_validity_s: float = 0.0
    dynamic_mva: float | None = None
    available_mva: float | None = None

    @property
    def derated(self) -> bool:
        """True when the dynamic ampacity is below the static nameplate."""
        return self.dynamic_ampacity_a < self.static_ampacity_a

    @property
    def overloaded(self) -> bool:
        """True when present current exceeds the dynamic ampacity."""
        return self.current_a > self.dynamic_ampacity_a


# ---------------------------------------------------------------------------
# Heat-balance physics (per metre of conductor)
# ---------------------------------------------------------------------------

def _air_properties(film_c: float) -> tuple[float, float, float]:
    """Thermal conductivity, kinematic viscosity, density of air at film temp."""
    k = 0.0242 + 7.2e-5 * film_c
    nu = 1.32e-5 + 9.5e-8 * film_c
    rho = 1.292 * 273.15 / (film_c + 273.15)
    return k, nu, rho


def wind_angle_factor(angle_deg: float) -> float:
    """IEEE 738 wind-direction factor (1.0 for crosswind, ~0.39 for parallel)."""
    phi = math.radians(angle_deg)
    return (1.194 - math.cos(phi)
            + 0.194 * math.cos(2.0 * phi)
            + 0.368 * math.sin(2.0 * phi))


def convection_loss_w_per_m(conductor_temp_c: float, ambient_c: float,
                            wind_mps: float, diameter_m: float,
                            wind_angle_deg: float = 90.0) -> float:
    """Convective heat loss rate (W/m); max of forced and natural convection."""
    dt = conductor_temp_c - ambient_c
    if dt <= 0.0 or diameter_m <= 0.0:
        return 0.0
    film = (conductor_temp_c + ambient_c) / 2.0
    k, nu, _ = _air_properties(film)
    re = max(0.0, wind_mps) * diameter_m / nu
    # IEEE 738 forced-convection terms; use the larger.
    q_low = (1.01 + 1.35 * re ** 0.52) * k * dt
    q_high = 0.754 * re ** 0.6 * k * dt
    q_forced = wind_angle_factor(wind_angle_deg) * max(q_low, q_high)
    # Natural convection: Churchill-Chu horizontal-cylinder correlation.
    beta = 1.0 / (film + 273.15)
    gr = 9.81 * beta * dt * diameter_m ** 3 / nu ** 2
    ra = gr * _AIR_PRANDTL
    nu_nat = 0.36 + 0.518 * ra ** 0.25
    q_natural = nu_nat * math.pi * k * dt
    return max(q_forced, q_natural)


def radiation_loss_w_per_m(conductor_temp_c: float, ambient_c: float,
                           diameter_m: float, emissivity: float) -> float:
    """Radiative heat loss rate (W/m)."""
    ts = conductor_temp_c + 273.15
    ta = ambient_c + 273.15
    return (math.pi * diameter_m * emissivity * _STEFAN_BOLTZMANN
            * (ts ** 4 - ta ** 4))


def solar_gain_w_per_m(solar_w_m2: float, diameter_m: float,
                       absorptivity: float) -> float:
    """Solar heat gain rate (W/m) on the projected area of the conductor."""
    return absorptivity * max(0.0, solar_w_m2) * diameter_m


def _resistance_per_m(conductor: LineConductor, temp_c: float) -> float:
    return conductor.r_ohm_per_m * (
        1.0 + conductor.temp_coeff_per_c * (temp_c - conductor.reference_temp_c))


def _cooling_w_per_m(conductor: LineConductor, weather: WeatherEvidence,
                     temp_c: float) -> float:
    return (convection_loss_w_per_m(temp_c, weather.ambient_c, weather.wind_mps,
                                     conductor.diameter_m, weather.wind_angle_deg)
            + radiation_loss_w_per_m(temp_c, weather.ambient_c,
                                     conductor.diameter_m, conductor.emissivity))


def dynamic_ampacity_a(conductor: LineConductor,
                       weather: WeatherEvidence) -> float:
    """Steady-state ampacity (A) under the given weather evidence.

    Solves the heat balance at the conductor's maximum allowable
    temperature: I = sqrt((cooling - solar gain) / R(Tmax)).
    """
    if weather.wind_mps < 0.0 or conductor.diameter_m <= 0.0:
        raise ValueError("invalid conductor or weather geometry")
    cooling = _cooling_w_per_m(conductor, weather, conductor.max_temp_c)
    gain = solar_gain_w_per_m(weather.solar_w_m2, conductor.diameter_m,
                              conductor.absorptivity)
    resistance = _resistance_per_m(conductor, conductor.max_temp_c)
    if resistance <= 0.0:
        raise ValueError("conductor resistance must be positive")
    if cooling <= gain:
        return 0.0
    return math.sqrt((cooling - gain) / resistance)


def conductor_temperature_c(conductor: LineConductor, weather: WeatherEvidence,
                            current_a: float) -> float:
    """Estimated steady-state conductor temperature (C) at the present load.

    Bounds the heat balance with bisection between the ambient temperature
    and ambient + 250 C.  Returns the upper bound if the heat balance
    cannot be reached (conductor overheated beyond the model range).
    """
    if current_a < 0.0:
        raise ValueError("current cannot be negative")
    gain = solar_gain_w_per_m(weather.solar_w_m2, conductor.diameter_m,
                              conductor.absorptivity)

    def balance(temp_c: float) -> float:
        heating = current_a ** 2 * _resistance_per_m(conductor, temp_c) + gain
        return heating - _cooling_w_per_m(conductor, weather, temp_c)

    lo = weather.ambient_c
    hi = weather.ambient_c + 250.0
    if balance(hi) >= 0.0:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if balance(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Rating reasons
# ---------------------------------------------------------------------------

def _derating_reasons(conductor: LineConductor, weather: WeatherEvidence,
                      dynamic_a: float) -> list[str]:
    """Explain why the dynamic ampacity is below/above the static basis."""
    reasons: list[str] = []
    if weather.ambient_c > _RATING_BASIS_TEMP_C + 1.0:
        reasons.append("high_ambient")
    if weather.wind_mps < 0.5:
        reasons.append("low_wind")
    if weather.solar_w_m2 >= 500.0:
        reasons.append("solar_heating")
    if weather.ambient_c < _RATING_BASIS_TEMP_C - 1.0:
        reasons.append("low_ambient")
    if weather.wind_mps >= 2.0:
        reasons.append("favorable_wind")
    if weather.solar_w_m2 <= 0.0:
        reasons.append("night_cooling")
    if dynamic_a < conductor.static_ampacity_a:
        reasons.append("derated")
    elif dynamic_a > conductor.static_ampacity_a:
        reasons.append("uplift")
    return reasons


def _fallback(conductor: LineConductor, current_a: float, now_s: float,
              reasons: tuple[str, ...],
              weather: WeatherEvidence | None = None) -> DynamicLineRating:
    """Build a static-fallback rating; dynamic margin is never reported."""
    kv = conductor.kv_ll
    mva = (math.sqrt(3.0) * kv * conductor.static_ampacity_a / 1000.0
           if kv else None)
    avail_mva = (math.sqrt(3.0) * kv * (conductor.static_ampacity_a - current_a)
                 / 1000.0 if kv else None)
    return DynamicLineRating(
        line_id=conductor.line_id,
        basis="static_fallback",
        static_ampacity_a=conductor.static_ampacity_a,
        dynamic_ampacity_a=conductor.static_ampacity_a,
        current_a=current_a,
        available_a=conductor.static_ampacity_a - current_a,
        valid_from_s=now_s,
        valid_until_s=now_s,
        reasons=reasons,
        ambient_c=weather.ambient_c if weather else None,
        wind_mps=weather.wind_mps if weather else None,
        solar_w_m2=weather.solar_w_m2 if weather else None,
        weather_age_s=(now_s - weather.timestamp_s) if weather else None,
        remaining_validity_s=0.0,
        dynamic_mva=mva,
        available_mva=avail_mva,
    )


def rate_line(conductor: LineConductor, weather: WeatherEvidence | None,
              current_a: float, now_s: float) -> DynamicLineRating:
    """Rate one line, falling back to static nameplate when weather evidence
    is missing, expired, or invalid.

    The returned dynamic values are computed; ``conductor.static_ampacity_a``
    is never modified.
    """
    if current_a < 0.0:
        raise ValueError("current cannot be negative")
    if weather is None:
        return _fallback(conductor, current_a, now_s, ("weather_missing",))

    age = now_s - weather.timestamp_s
    if age > weather.valid_for_s:
        return _fallback(conductor, current_a, now_s, ("weather_expired",),
                         weather)
    if weather.quality not in ("good", "suspect"):
        return _fallback(conductor, current_a, now_s, ("weather_invalid",),
                         weather)

    dynamic_a = dynamic_ampacity_a(conductor, weather)
    temp_c = conductor_temperature_c(conductor, weather, current_a)
    reasons = _derating_reasons(conductor, weather, dynamic_a)
    if current_a > dynamic_a:
        reasons.append("overload")

    kv = conductor.kv_ll
    dynamic_mva = (math.sqrt(3.0) * kv * dynamic_a / 1000.0) if kv else None
    available_mva = (math.sqrt(3.0) * kv * (dynamic_a - current_a) / 1000.0
                     if kv else None)
    return DynamicLineRating(
        line_id=conductor.line_id,
        basis="dynamic",
        static_ampacity_a=conductor.static_ampacity_a,
        dynamic_ampacity_a=dynamic_a,
        current_a=current_a,
        available_a=dynamic_a - current_a,
        valid_from_s=weather.timestamp_s,
        valid_until_s=weather.timestamp_s + weather.valid_for_s,
        reasons=tuple(reasons),
        conductor_temp_c=temp_c,
        ambient_c=weather.ambient_c,
        wind_mps=weather.wind_mps,
        solar_w_m2=weather.solar_w_m2,
        weather_age_s=age,
        remaining_validity_s=weather.valid_for_s - age,
        dynamic_mva=dynamic_mva,
        available_mva=available_mva,
    )


# ---------------------------------------------------------------------------
# Batch rating
# ---------------------------------------------------------------------------

def rate_lines(jobs: list[tuple[LineConductor, WeatherEvidence | None, float]],
               now_s: float) -> list[DynamicLineRating]:
    """Rate many lines in one batch.

    Each job is ``(conductor, weather_evidence, current_a)``.  Every
    returned rating carries its own basis, validity window, and derating /
    fallback reasons so batch results remain auditable.
    """
    return [rate_line(conductor, weather, current_a, now_s)
            for conductor, weather, current_a in jobs]


def summarize_batch(results: list[DynamicLineRating]) -> dict:
    """Aggregate a batch of ratings for reporting.

    Counts dynamic vs static-fallback bases, deratings, and overloads,
    and lists the line ids in each category so operators can see why any
    line was derated or fell back.
    """
    fallback: dict[str, int] = {}
    derated: list[str] = []
    overloaded: list[str] = []
    uplift: list[str] = []
    for r in results:
        if r.basis == "static_fallback":
            for reason in r.reasons:
                fallback[reason] = fallback.get(reason, 0) + 1
        if r.derated:
            derated.append(r.line_id)
        if r.overloaded:
            overloaded.append(r.line_id)
        if r.basis == "dynamic" and not r.derated and not r.overloaded:
            uplift.append(r.line_id)
    return {
        "total": len(results),
        "dynamic": sum(1 for r in results if r.basis == "dynamic"),
        "static_fallback": sum(1 for r in results
                               if r.basis == "static_fallback"),
        "derated": derated,
        "overloaded": overloaded,
        "uplift": uplift,
        "fallback_reasons": fallback,
    }


def from_catalog(code: str, diameter_m: float, max_temp_c: float = 75.0,
                 kv_ll: float | None = None,
                 line_id: str | None = None) -> LineConductor:
    """Build a LineConductor from the static conductor catalog.

    The catalog's nameplate ampacity is copied, never referenced by
    dynamic calculations.
    """
    catalog = get_conductor(code)
    return LineConductor(
        line_id=line_id or code,
        r_ohm_per_m=catalog.r_ohm_km / 1000.0,
        diameter_m=diameter_m,
        static_ampacity_a=catalog.ampacity_a,
        max_temp_c=max_temp_c,
        kv_ll=kv_ll,
    )
