from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

SQRT3 = math.sqrt(3.0)
STEFAN_BOLTZMANN = 5.670374419e-8
GRAVITY = 9.80665


@dataclass(frozen=True)
class WeatherEvidence:
    """Timestamped weather evidence used for a dynamic line rating."""

    ambient_temp_c: float
    wind_speed_mps: float
    solar_irradiance_w_m2: float
    observed_at_s: float = 0.0
    quality: str = "good"


@dataclass(frozen=True)
class LineRatingRequest:
    """Static conductor data plus the operating state for one line."""

    line_id: str
    static_ampacity_a: float
    current_a: float = 0.0
    diameter_mm: float = 18.0
    max_conductor_temp_c: float = 75.0
    kv_ll: float | None = None
    emissivity: float = 0.8
    absorptivity: float = 0.8
    weather: WeatherEvidence | None = None
    now_s: float | None = None
    weather_ttl_s: float = 900.0
    reference_ambient_temp_c: float = 40.0
    reference_wind_speed_mps: float = 1.0
    reference_solar_w_m2: float = 1000.0

    @classmethod
    def for_conductor(
        cls,
        line_id: str,
        conductor: object,
        current_a: float = 0.0,
        weather: WeatherEvidence | None = None,
        now_s: float | None = None,
        weather_ttl_s: float = 900.0,
        kv_ll: float | None = None,
    ) -> "LineRatingRequest":
        return cls(
            line_id=line_id,
            static_ampacity_a=conductor.ampacity_a,
            current_a=current_a,
            diameter_mm=getattr(conductor, "diameter_mm", 18.0),
            max_conductor_temp_c=getattr(conductor, "max_temp_c", 75.0),
            kv_ll=kv_ll,
            weather=weather,
            now_s=now_s,
            weather_ttl_s=weather_ttl_s,
        )


@dataclass(frozen=True)
class DynamicLineCapacity:
    """A time-limited operating capacity; static conductor ratings are not modified."""

    line_id: str
    rating_source: str
    derated: bool
    ampacity_a: float
    static_ampacity_a: float
    current_a: float
    available_a: float
    headroom_a: float
    utilization_pu: float
    overloaded: bool
    reasons: tuple[str, ...]
    fallback_reasons: tuple[str, ...]
    derating_reasons: tuple[str, ...]
    observed_at_s: float | None
    expires_at_s: float | None
    weather: WeatherEvidence | None
    available_mva: float | None
    headroom_mva: float | None

    @property
    def dynamic_ampacity_a(self) -> float:
        return self.ampacity_a

    @property
    def available_capacity_a(self) -> float:
        return self.available_a

    @property
    def valid_until_s(self) -> float | None:
        return self.expires_at_s

    @property
    def fallback(self) -> bool:
        return self.rating_source == "static"


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_request(request: LineRatingRequest) -> None:
    positive_fields = {
        "static_ampacity_a": request.static_ampacity_a,
        "diameter_mm": request.diameter_mm,
        "max_conductor_temp_c": request.max_conductor_temp_c,
        "weather_ttl_s": request.weather_ttl_s,
    }
    for name, value in positive_fields.items():
        if not _finite_number(value) or value <= 0:
            raise ValueError(f"{name} must be positive")
    if not _finite_number(request.current_a) or request.current_a < 0:
        raise ValueError("current_a must be nonnegative")
    if (
        not _finite_number(request.emissivity)
        or not _finite_number(request.absorptivity)
        or not 0.0 <= request.emissivity <= 1.0
        or not 0.0 <= request.absorptivity <= 1.0
    ):
        raise ValueError("emissivity and absorptivity must be between 0 and 1")
    if not _finite_number(request.max_conductor_temp_c):
        raise ValueError("max conductor temperature must be finite")
    if (
        not _finite_number(request.reference_ambient_temp_c)
        or not _finite_number(request.reference_wind_speed_mps)
        or not _finite_number(request.reference_solar_w_m2)
    ):
        raise ValueError("reference weather values must be finite")
    if request.reference_ambient_temp_c >= request.max_conductor_temp_c:
        raise ValueError("reference ambient temperature must be below conductor limit")
    if request.reference_wind_speed_mps < 0 or request.reference_solar_w_m2 < 0:
        raise ValueError("reference weather values must be nonnegative")


def _air_properties(film_temp_c: float) -> tuple[float, float, float]:
    # Standard-air approximations around distribution-conductor temperatures.
    conductivity = 2.424e-2 + 7.477e-5 * film_temp_c - 4.409e-9 * film_temp_c**2
    kinematic_viscosity = 1.320e-5 + 9.500e-8 * film_temp_c
    prandtl = 0.71
    return conductivity, max(kinematic_viscosity, 1e-9), prandtl


def _convection_w_m(
    diameter_m: float,
    conductor_temp_c: float,
    ambient_temp_c: float,
    wind_speed_mps: float,
) -> float:
    delta = conductor_temp_c - ambient_temp_c
    if delta <= 0:
        return 0.0
    film_temp = (conductor_temp_c + ambient_temp_c) / 2.0
    conductivity, viscosity, prandtl = _air_properties(film_temp)

    reynolds = max(0.0, wind_speed_mps) * diameter_m / viscosity
    forced_nusselt = 0.3 + 0.62 * math.sqrt(reynolds) * prandtl ** (1.0 / 3.0)
    forced = math.pi * conductivity * forced_nusselt * delta

    beta = 1.0 / (film_temp + 273.15)
    rayleigh = GRAVITY * beta * delta * diameter_m**3 / viscosity**2 * prandtl
    natural_term = (
        0.387
        * rayleigh ** (1.0 / 6.0)
        / (1.0 + (0.559 / prandtl) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    )
    natural_nusselt = (0.60 + natural_term) ** 2
    natural = math.pi * conductivity * natural_nusselt * delta
    return max(forced, natural)


def _radiation_w_m(
    diameter_m: float,
    conductor_temp_c: float,
    ambient_temp_c: float,
    emissivity: float,
) -> float:
    conductor_k = conductor_temp_c + 273.15
    ambient_k = ambient_temp_c + 273.15
    return (
        emissivity
        * STEFAN_BOLTZMANN
        * math.pi
        * diameter_m
        * (conductor_k**4 - ambient_k**4)
    )


def _net_cooling_w_m(
    request: LineRatingRequest, ambient_c: float, wind_mps: float, solar_w_m2: float
) -> float:
    diameter_m = request.diameter_mm / 1000.0
    convection = _convection_w_m(
        diameter_m, request.max_conductor_temp_c, ambient_c, wind_mps
    )
    radiation = _radiation_w_m(
        diameter_m, request.max_conductor_temp_c, ambient_c, request.emissivity
    )
    solar_gain = request.absorptivity * solar_w_m2 * diameter_m
    return convection + radiation - solar_gain


def _ampacity_a(
    request: LineRatingRequest, ambient_c: float, wind_mps: float, solar_w_m2: float
) -> float:
    net = _net_cooling_w_m(request, ambient_c, wind_mps, solar_w_m2)
    if net <= 0.0:
        return 0.0
    reference_net = _net_cooling_w_m(
        request,
        request.reference_ambient_temp_c,
        request.reference_wind_speed_mps,
        request.reference_solar_w_m2,
    )
    if reference_net <= 0.0:
        raise ValueError("static rating is inconsistent with reference weather")
    return request.static_ampacity_a * math.sqrt(net / reference_net)


def _mva(kv_ll: float | None, current_a: float) -> float | None:
    if kv_ll is None:
        return None
    return SQRT3 * kv_ll * current_a / 1000.0


def _capacity_result(
    request: LineRatingRequest,
    ampacity_a: float,
    source: str,
    reasons: tuple[str, ...],
    fallback_reasons: tuple[str, ...],
    derating_reasons: tuple[str, ...],
    weather: WeatherEvidence | None,
    observed_at_s: float | None,
    expires_at_s: float | None,
) -> DynamicLineCapacity:
    headroom_a = ampacity_a - request.current_a
    available_a = max(0.0, headroom_a)
    headroom_mva = _mva(request.kv_ll, headroom_a)
    available_mva = None if headroom_mva is None else max(0.0, headroom_mva)
    if ampacity_a <= 0.0:
        utilization = math.inf if request.current_a > 0.0 else 0.0
    else:
        utilization = request.current_a / ampacity_a

    all_reasons = list(reasons)
    if headroom_a < -1e-9:
        all_reasons.append("current_overload")
    return DynamicLineCapacity(
        line_id=request.line_id,
        rating_source=source,
        derated=bool(derating_reasons),
        ampacity_a=ampacity_a,
        static_ampacity_a=request.static_ampacity_a,
        current_a=request.current_a,
        available_a=available_a,
        headroom_a=headroom_a,
        utilization_pu=utilization,
        overloaded=headroom_a < -1e-9,
        reasons=tuple(all_reasons),
        fallback_reasons=fallback_reasons,
        derating_reasons=derating_reasons,
        observed_at_s=observed_at_s,
        expires_at_s=expires_at_s,
        weather=weather,
        available_mva=available_mva,
        headroom_mva=headroom_mva,
    )


def dynamic_line_capacity(
    request: LineRatingRequest, now_s: float | None = None
) -> DynamicLineCapacity:
    """Calculate a weather-derived capacity, falling back when evidence is unusable."""

    _validate_request(request)
    calculation_time = request.now_s if now_s is None else now_s
    if calculation_time is None:
        calculation_time = (
            request.weather.observed_at_s if request.weather is not None else 0.0
        )
    if not _finite_number(calculation_time):
        raise ValueError("calculation time must be finite")

    ttl = request.weather_ttl_s
    weather = request.weather
    if weather is not None:
        weather = WeatherEvidence(
            ambient_temp_c=weather.ambient_temp_c,
            wind_speed_mps=weather.wind_speed_mps,
            solar_irradiance_w_m2=weather.solar_irradiance_w_m2,
            observed_at_s=weather.observed_at_s,
            quality=weather.quality,
        )

    if weather is None:
        valid, state_reason = False, "weather_evidence_missing"
    else:
        values = (
            weather.ambient_temp_c,
            weather.wind_speed_mps,
            weather.solar_irradiance_w_m2,
            weather.observed_at_s,
        )
        if not all(_finite_number(value) for value in values):
            valid, state_reason = False, "weather_evidence_invalid_value"
        elif weather.quality not in {"good", "suspect"}:
            valid, state_reason = False, "weather_evidence_invalid_quality"
        elif weather.wind_speed_mps < 0 or weather.solar_irradiance_w_m2 < 0:
            valid, state_reason = False, "weather_evidence_invalid_value"
        elif weather.observed_at_s > calculation_time:
            valid, state_reason = False, "weather_evidence_future"
        elif calculation_time - weather.observed_at_s > ttl:
            valid, state_reason = False, "weather_evidence_expired"
        else:
            valid, state_reason = True, None

    if not valid:
        return _capacity_result(
            request,
            request.static_ampacity_a,
            "static",
            (state_reason,),
            (state_reason,),
            (),
            None,
            None,
            None,
        )

    dynamic_a = _ampacity_a(
        request,
        weather.ambient_temp_c,
        weather.wind_speed_mps,
        weather.solar_irradiance_w_m2,
    )
    factors: list[str] = []
    if dynamic_a < request.static_ampacity_a - 1e-9:
        if (
            _ampacity_a(
                request,
                request.reference_ambient_temp_c,
                weather.wind_speed_mps,
                weather.solar_irradiance_w_m2,
            )
            > dynamic_a + 1e-9
        ):
            factors.append("high_ambient_temperature")
        if (
            _ampacity_a(
                request,
                weather.ambient_temp_c,
                request.reference_wind_speed_mps,
                weather.solar_irradiance_w_m2,
            )
            > dynamic_a + 1e-9
        ):
            factors.append("low_wind_speed")
        if (
            _ampacity_a(
                request,
                weather.ambient_temp_c,
                weather.wind_speed_mps,
                request.reference_solar_w_m2,
            )
            > dynamic_a + 1e-9
        ):
            factors.append("high_solar_heating")

    expires_at_s = weather.observed_at_s + ttl
    return _capacity_result(
        request,
        dynamic_a,
        "dynamic_weather",
        tuple(factors),
        (),
        tuple(factors),
        weather,
        weather.observed_at_s,
        expires_at_s,
    )


def batch_dynamic_line_capacities(
    requests: Iterable[LineRatingRequest], now_s: float | None = None
) -> tuple[DynamicLineCapacity, ...]:
    """Calculate capacities for many lines without changing their static ratings."""

    return tuple(dynamic_line_capacity(request, now_s) for request in requests)


def dynamic_rating(
    line_id: str,
    conductor: object,
    current_a: float,
    weather: WeatherEvidence | None = None,
    now_s: float | None = None,
    weather_ttl_s: float = 900.0,
    kv_ll: float | None = None,
) -> DynamicLineCapacity:
    """Convenience wrapper for a conductor from ``relaygrid.conductors.catalog``."""

    request = LineRatingRequest.for_conductor(
        line_id,
        conductor,
        current_a=current_a,
        weather=weather,
        now_s=now_s,
        weather_ttl_s=weather_ttl_s,
        kv_ll=kv_ll,
    )
    return dynamic_line_capacity(request, now_s)


# Domain language uses both "evidence" and "observation"; expose both spellings.
WeatherObservation = WeatherEvidence
LineCapacity = DynamicLineCapacity


# Backwards-friendly descriptive aliases.
calculate_dynamic_line_capacity = dynamic_line_capacity
dynamic_line_rating = dynamic_line_capacity
batch_dynamic_line_ratings = batch_dynamic_line_capacities
