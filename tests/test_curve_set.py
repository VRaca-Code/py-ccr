"""Tests for multi-curve market-data containers."""

import pytest

from pyccr.market.curve_set import CurveSet
from pyccr.market.discount_curve import DiscountCurve


@pytest.fixture
def curves() -> CurveSet:
    discount = DiscountCurve(
        tenors=(1.0, 5.0, 10.0),
        zero_rates=(0.025, 0.028, 0.030),
    )

    projection = DiscountCurve(
        tenors=(1.0, 5.0, 10.0),
        zero_rates=(0.028, 0.031, 0.033),
    )

    return CurveSet(
        discount_curves={"EUR": discount},
        projection_curves={("EUR", "EURIBOR6M"): projection},
    )


def test_discount_curve_lookup(curves: CurveSet) -> None:
    assert curves.discount_curve("eur") is curves.discount_curves["EUR"]


def test_projection_curve_lookup(curves: CurveSet) -> None:
    assert (
        curves.projection_curve("eur", "euribor6m")
        is curves.projection_curves[("EUR", "EURIBOR6M")]
    )


def test_missing_discount_curve(curves: CurveSet) -> None:
    with pytest.raises(KeyError):
        curves.discount_curve("USD")


def test_missing_projection_curve(curves: CurveSet) -> None:
    with pytest.raises(KeyError):
        curves.projection_curve("EUR", "EURIBOR3M")


def test_discount_curve_is_immutable(curves: CurveSet) -> None:
    with pytest.raises(TypeError):
        curves.discount_curves["USD"] = curves.discount_curves["EUR"]


def test_projection_curve_is_immutable(curves: CurveSet) -> None:
    with pytest.raises(TypeError):
        curves.projection_curves[("USD", "SOFR")] = curves.projection_curves[
            ("EUR", "EURIBOR6M")
        ]


def test_normalized_currency_keys() -> None:
    curve = DiscountCurve(
        tenors=(1.0, 5.0),
        zero_rates=(0.02, 0.03),
    )

    curves = CurveSet(
        discount_curves={" eur ": curve},
        projection_curves={(" eur ", " euribor6m "): curve},
    )

    assert curves.discount_curve("EUR") is curve
    assert curves.projection_curve("EUR", "EURIBOR6M") is curve


def test_duplicate_normalized_currency_rejected() -> None:
    curve = DiscountCurve(
        tenors=(1.0, 5.0),
        zero_rates=(0.02, 0.03),
    )

    with pytest.raises(ValueError):
        CurveSet(
            discount_curves={"EUR": curve, "eur": curve},
            projection_curves={},
        )


def test_invalid_curve_type() -> None:
    with pytest.raises(TypeError):
        CurveSet(
            discount_curves={"EUR": "not a curve"},
            projection_curves={},
        )


def test_invalid_projection_key() -> None:
    curve = DiscountCurve(
        tenors=(1.0, 5.0),
        zero_rates=(0.02, 0.03),
    )

    with pytest.raises(ValueError):
        CurveSet(
            discount_curves={},
            projection_curves={"EURIBOR6M": curve},
        )
