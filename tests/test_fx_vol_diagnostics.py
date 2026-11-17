"""Regression tests for FX volatility-surface diagnostics."""

import numpy as np
import pytest

from pyccr.market.fx_vol_diagnostics import (
    CalendarVarianceViolation,
    calendar_variance_violations,
)
from pyccr.market.fx_vol_surface import FXVolSurface


def make_surface(
    maturities=(1.0, 2.0, 3.0),
    strikes=(0.9, 1.1),
    volatilities=(
        (0.20, 0.25),
        (0.20, 0.25),
        (0.20, 0.25),
    ),
):
    return FXVolSurface(
        maturities=maturities,
        strikes=strikes,
        volatilities=volatilities,
    )


def test_constant_volatility_surface_has_no_violations():
    surface = make_surface()
    assert calendar_variance_violations(surface) == ()


def test_increasing_total_variance_has_no_violations():
    surface = make_surface(
        maturities=(1.0, 2.0, 3.0),
        strikes=(1.0,),
        volatilities=((0.10,), (0.15,), (0.20,)),
    )

    assert calendar_variance_violations(surface) == ()


def test_equal_total_variance_has_no_violations():
    surface = make_surface(
        maturities=(1.0, 2.0),
        strikes=(1.0,),
        volatilities=((0.20,), (0.20 / np.sqrt(2.0),)),
    )

    assert calendar_variance_violations(surface) == ()


def test_decreasing_total_variance_detected():
    surface = make_surface(
        maturities=(1.0, 2.0),
        strikes=(1.0,),
        volatilities=((0.30,), (0.10,)),
    )

    violations = calendar_variance_violations(surface)

    assert len(violations) == 1
    violation = violations[0]

    assert isinstance(violation, CalendarVarianceViolation)
    assert violation.earlier_maturity == 1.0
    assert violation.later_maturity == 2.0
    assert violation.strike == 1.0
    assert violation.earlier_variance == pytest.approx(0.09)
    assert violation.later_variance == pytest.approx(0.02)
    assert violation.variance_change == pytest.approx(-0.07)


def test_multiple_strikes_and_maturity_intervals():
    surface = make_surface(
        maturities=(1.0, 2.0, 3.0),
        strikes=(0.9, 1.1),
        volatilities=(
            (0.30, 0.10),
            (0.10, 0.30),
            (0.30, 0.10),
        ),
    )

    violations = calendar_variance_violations(surface)

    assert len(violations) == 2

    assert (
        violations[0].earlier_maturity,
        violations[0].later_maturity,
        violations[0].strike,
    ) == (1.0, 2.0, 0.9)

    assert (
        violations[1].earlier_maturity,
        violations[1].later_maturity,
        violations[1].strike,
    ) == (2.0, 3.0, 1.1)


@pytest.mark.parametrize("tolerance", [0.0, 1e-12, 0.01])
def test_violation_detected_below_tolerance_threshold(tolerance):
    surface = make_surface(
        maturities=(1.0, 2.0),
        strikes=(1.0,),
        volatilities=((0.30,), (0.10,)),
    )

    assert len(calendar_variance_violations(surface, tolerance=tolerance)) == 1


def test_violation_suppressed_by_large_tolerance():
    surface = make_surface(
        maturities=(1.0, 2.0),
        strikes=(1.0,),
        volatilities=((0.30,), (0.10,)),
    )

    assert (
        calendar_variance_violations(
            surface,
            tolerance=0.07,
        )
        == ()
    )


@pytest.mark.parametrize(
    "tolerance",
    [-1.0, np.nan, np.inf, -np.inf],
)
def test_invalid_tolerance_rejected(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        calendar_variance_violations(
            make_surface(),
            tolerance=tolerance,
        )


def test_single_maturity_has_no_adjacent_intervals():
    surface = make_surface(
        maturities=(1.0,),
        strikes=(1.0,),
        volatilities=((0.20,),),
    )

    assert calendar_variance_violations(surface) == ()


def test_zero_volatility_surface_has_no_violations():
    surface = make_surface(
        maturities=(1.0, 2.0),
        strikes=(1.0,),
        volatilities=((0.0,), (0.0,)),
    )

    assert calendar_variance_violations(surface) == ()


def test_surface_remains_unchanged():
    surface = make_surface(
        maturities=(1.0, 2.0),
        strikes=(1.0,),
        volatilities=((0.30,), (0.10,)),
    )

    before = (
        surface.maturities,
        surface.strikes,
        surface.volatilities,
    )

    calendar_variance_violations(surface)

    after = (
        surface.maturities,
        surface.strikes,
        surface.volatilities,
    )

    assert after == before


def test_violation_records_are_immutable():
    violation = CalendarVarianceViolation(
        earlier_maturity=1.0,
        later_maturity=2.0,
        strike=1.0,
        earlier_variance=0.09,
        later_variance=0.02,
    )

    with pytest.raises(AttributeError):
        violation.strike = 1.2
