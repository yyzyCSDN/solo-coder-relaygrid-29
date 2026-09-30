from __future__ import annotations

import pytest

from relaygrid.conductors.catalog import CATALOG, get
from relaygrid.conductors.dynamic_rating import (
    DynamicLineCapacity,
    LineRatingRequest,
    WeatherEvidence,
    batch_dynamic_line_capacities,
    dynamic_line_capacity,
)


def request(line_id: str = "fdr-1", current_a: float = 300.0, **kwargs):
    weather = kwargs.pop("weather", WeatherEvidence(35.0, 2.0, 800.0, 100.0, "good"))
    now_s = kwargs.pop("now_s", 100.0)
    weather_ttl_s = kwargs.pop("weather_ttl_s", 900.0)
    return LineRatingRequest(
        line_id,
        340.0,
        current_a=current_a,
        diameter_mm=14.3,
        weather=weather,
        now_s=now_s,
        weather_ttl_s=weather_ttl_s,
        kv_ll=12.47,
        **kwargs,
    )


def test_favorable_weather_has_time_limited_capacity():
    result = dynamic_line_capacity(request())

    assert result.rating_source == "dynamic_weather"
    assert result.ampacity_a > 340.0
    assert result.derating_reasons == ()
    assert result.available_a == pytest.approx(result.ampacity_a - 300.0)
    assert result.observed_at_s == 100.0
    assert result.expires_at_s == 1000.0
    assert result.available_mva == pytest.approx(
        1.7320508075688772 * 12.47 * result.available_a / 1000.0
    )


def test_missing_and_expired_weather_fall_back_to_static_rating():
    missing = dynamic_line_capacity(request(weather=None, weather_ttl_s=900.0))
    expired = dynamic_line_capacity(
        request(weather=WeatherEvidence(35.0, 2.0, 800.0, 0.0), now_s=901.0)
    )

    assert missing.rating_source == "static"
    assert missing.ampacity_a == 340.0
    assert missing.reasons == ("weather_evidence_missing",)
    assert missing.expires_at_s is None

    assert expired.rating_source == "static"
    assert expired.ampacity_a == 340.0
    assert expired.reasons == ("weather_evidence_expired",)
    assert expired.fallback_reasons == ("weather_evidence_expired",)
    assert expired.derating_reasons == ()


def test_bad_quality_weather_falls_back_to_static_rating():
    result = dynamic_line_capacity(
        request(weather=WeatherEvidence(35.0, 2.0, 800.0, 100.0, "bad"))
    )

    assert result.rating_source == "static"
    assert result.ampacity_a == 340.0
    assert result.fallback_reasons == ("weather_evidence_invalid_quality",)


def test_hot_calm_weather_marks_each_derating_cause_and_overload():
    result = dynamic_line_capacity(
        request(
            current_a=320.0,
            weather=WeatherEvidence(50.0, 0.0, 1000.0, 100.0),
        )
    )

    assert result.rating_source == "dynamic_weather"
    assert result.ampacity_a < 340.0
    assert result.derating_reasons == (
        "high_ambient_temperature",
        "low_wind_speed",
    )
    assert result.overloaded
    assert result.available_a == 0.0
    assert result.headroom_a < 0.0
    assert result.reasons[-1] == "current_overload"


def test_batch_calculation_tags_results_and_preserves_static_catalog_rating():
    conductor = get("ACSR_4_0")
    requests = [
        LineRatingRequest.for_conductor(
            "good", conductor, 300.0, WeatherEvidence(30.0, 3.0, 600.0, 0.0), 0.0
        ),
        LineRatingRequest.for_conductor(
            "hot", conductor, 300.0, WeatherEvidence(50.0, 0.0, 1000.0, 0.0), 0.0
        ),
        LineRatingRequest.for_conductor("missing", conductor, 300.0, None, 0.0),
    ]

    results = batch_dynamic_line_capacities(requests)

    assert len(results) == 3
    assert all(isinstance(item, DynamicLineCapacity) for item in results)
    assert results[0].rating_source == "dynamic_weather"
    assert results[1].derating_reasons
    assert results[2].rating_source == "static"
    assert conductor.ampacity_a == 340.0
    assert CATALOG["ACSR_4_0"].ampacity_a == 340.0
    assert requests[1].static_ampacity_a == 340.0


def test_catalog_conductor_defaults_are_used():
    result = dynamic_line_capacity(
        LineRatingRequest.for_conductor(
            "cat",
            get("ACSR_4_0"),
            300.0,
            WeatherEvidence(35.0, 2.0, 800.0, 0.0),
            now_s=0.0,
            kv_ll=12.47,
        )
    )

    assert result.rating_source == "dynamic_weather"
    assert result.available_mva is not None
