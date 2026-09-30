from __future__ import annotations

import pytest

from relaygrid.calculators.dynamic_line_rating import (
    DynamicLineRating,
    LineConductor,
    WeatherEvidence,
    conductor_temperature_c,
    dynamic_ampacity_a,
    from_catalog,
    rate_line,
    rate_lines,
    summarize_batch,
)
from relaygrid.conductors.catalog import CATALOG

# 336.4 kcmil AAAC equivalent: ~1 in diameter, 0.161 ohm/km at 25 C.
LINE = LineConductor(
    line_id="L1",
    r_ohm_per_m=0.161e-3,
    diameter_m=0.0254,
    static_ampacity_a=420.0,
    kv_ll=12.47,
)
NOW = 1_000_000.0


def weather(ambient=25.0, wind=0.6, solar=1000.0, age_s=0.0,
            valid_for_s=1800.0, quality="good", angle=90.0):
    return WeatherEvidence(
        ambient_c=ambient, wind_mps=wind, solar_w_m2=solar,
        timestamp_s=NOW - age_s, valid_for_s=valid_for_s,
        wind_angle_deg=angle, quality=quality,
    )


def test_night_windy_cold_uplifts_above_nameplate():
    w = weather(ambient=5.0, wind=3.0, solar=0.0)
    r = rate_line(LINE, w, current_a=200.0, now_s=NOW)
    assert r.basis == "dynamic"
    assert r.dynamic_ampacity_a > LINE.static_ampacity_a
    assert r.available_a > LINE.static_ampacity_a - 200.0
    assert "low_ambient" in r.reasons
    assert "favorable_wind" in r.reasons
    assert "night_cooling" in r.reasons
    assert "uplift" in r.reasons
    assert not r.derated


def test_hot_calm_sun_derates_below_nameplate():
    w = weather(ambient=40.0, wind=0.0, solar=1000.0)
    r = rate_line(LINE, w, current_a=200.0, now_s=NOW)
    assert r.basis == "dynamic"
    assert r.dynamic_ampacity_a < LINE.static_ampacity_a
    assert r.derated
    for reason in ("high_ambient", "low_wind", "solar_heating", "derated"):
        assert reason in r.reasons


def test_night_rating_exceeds_day_rating():
    day = weather(ambient=30.0, wind=0.6, solar=1000.0)
    night = weather(ambient=30.0, wind=0.6, solar=0.0)
    assert dynamic_ampacity_a(LINE, night) > dynamic_ampacity_a(LINE, day)


def test_wind_cools_conductor():
    calm = weather(ambient=35.0, wind=0.0, solar=800.0)
    breezy = weather(ambient=35.0, wind=3.0, solar=800.0)
    assert dynamic_ampacity_a(LINE, breezy) > dynamic_ampacity_a(LINE, calm)


def test_conductor_temperature_rises_with_current():
    w = weather(ambient=25.0, wind=0.6, solar=1000.0)
    t_low = conductor_temperature_c(LINE, w, current_a=100.0)
    t_high = conductor_temperature_c(LINE, w, current_a=500.0)
    assert t_high > t_low
    assert t_low >= w.ambient_c


def test_overload_flagged_when_current_exceeds_dynamic():
    w = weather(ambient=45.0, wind=0.0, solar=1000.0)
    r = rate_line(LINE, w, current_a=600.0, now_s=NOW)
    assert r.overloaded
    assert "overload" in r.reasons
    assert r.available_a < 0.0


def test_missing_weather_falls_back_to_static():
    r = rate_line(LINE, None, current_a=200.0, now_s=NOW)
    assert r.basis == "static_fallback"
    assert r.dynamic_ampacity_a == LINE.static_ampacity_a
    assert r.available_a == LINE.static_ampacity_a - 200.0
    assert r.conductor_temp_c is None
    assert r.ambient_c is None and r.wind_mps is None
    assert r.reasons == ("weather_missing",)
    assert r.valid_from_s == NOW and r.valid_until_s == NOW
    assert r.remaining_validity_s == 0.0


def test_expired_weather_falls_back_to_static():
    w = weather(age_s=1801.0, valid_for_s=1800.0)
    r = rate_line(LINE, w, current_a=200.0, now_s=NOW)
    assert r.basis == "static_fallback"
    assert r.reasons == ("weather_expired",)
    assert r.weather_age_s == 1801.0
    assert r.dynamic_ampacity_a == LINE.static_ampacity_a


def test_bad_quality_weather_falls_back():
    w = weather(quality="bad")
    r = rate_line(LINE, w, current_a=200.0, now_s=NOW)
    assert r.basis == "static_fallback"
    assert r.reasons == ("weather_invalid",)


def test_suspect_quality_still_used():
    w = weather(quality="suspect")
    r = rate_line(LINE, w, current_a=200.0, now_s=NOW)
    assert r.basis == "dynamic"


def test_validity_window_reflects_evidence():
    w = weather(age_s=600.0, valid_for_s=1800.0)
    r = rate_line(LINE, w, current_a=200.0, now_s=NOW)
    assert r.basis == "dynamic"
    assert r.valid_from_s == w.timestamp_s
    assert r.valid_until_s == w.timestamp_s + 1800.0
    assert r.remaining_validity_s == 1200.0


def test_mva_values_present_with_voltage():
    w = weather()
    r = rate_line(LINE, w, current_a=200.0, now_s=NOW)
    assert r.dynamic_mva is not None and r.dynamic_mva > 0
    assert r.available_mva is not None
    assert abs(r.dynamic_mva - (r.available_mva
              + (3 ** 0.5) * LINE.kv_ll * 200.0 / 1000.0)) < 1e-9


def test_batch_marks_each_rating_with_reasons():
    jobs = [
        (LINE, weather(ambient=40.0, wind=0.0, solar=1000.0), 200.0),
        (LINE, None, 200.0),
        (LINE, weather(age_s=9999.0), 200.0),
        (LINE, weather(ambient=5.0, wind=4.0, solar=0.0), 200.0),
    ]
    results = rate_lines(jobs, now_s=NOW)
    assert len(results) == 4
    assert results[0].basis == "dynamic" and results[0].derated
    assert "high_ambient" in results[0].reasons
    assert results[1].basis == "static_fallback"
    assert results[2].reasons == ("weather_expired",)
    assert results[3].basis == "dynamic" and not results[3].derated
    summary = summarize_batch(results)
    assert summary["total"] == 4
    assert summary["dynamic"] == 2
    assert summary["static_fallback"] == 2
    assert summary["derated"] == ["L1"]
    assert summary["fallback_reasons"] == {"weather_missing": 1,
                                           "weather_expired": 1}


def test_nameplate_never_mutated_by_rating():
    before = LINE.static_ampacity_a
    catalog_before = {code: c.ampacity_a for code, c in CATALOG.items()}
    rate_line(LINE, weather(ambient=45.0, wind=0.0, solar=1000.0), 900.0, NOW)
    rate_line(LINE, None, 900.0, NOW)
    assert LINE.static_ampacity_a == before
    assert {code: c.ampacity_a for code, c in CATALOG.items()} == catalog_before


def test_from_catalog_uses_nameplate_without_writing():
    c = from_catalog("AAAC_336", diameter_m=0.0254, kv_ll=12.47)
    assert c.static_ampacity_a == CATALOG["AAAC_336"].ampacity_a
    w = weather(ambient=40.0, wind=0.0, solar=1000.0)
    r = rate_line(c, w, current_a=200.0, now_s=NOW)
    assert r.line_id == "AAAC_336"
    assert c.static_ampacity_a == CATALOG["AAAC_336"].ampacity_a


def test_negative_current_rejected():
    with pytest.raises(ValueError):
        rate_line(LINE, weather(), current_a=-1.0, now_s=NOW)
    with pytest.raises(ValueError):
        conductor_temperature_c(LINE, weather(), current_a=-1.0)
