"""Regression tests for the FX implied-volatility surface."""

import numpy as np
import pytest

from pyccr.market.fx_vol_surface import FXVolSurface


@pytest.fixture
def surface():
    return FXVolSurface(
        maturities=(1.0, 2.0),
        strikes=(0.9, 1.1),
        volatilities=(
            (0.10, 0.20),
            (0.15, 0.25),
        ),
    )


@pytest.mark.parametrize(
    ("maturity", "strike", "expected"),
    [
        (1.0, 0.9, 0.10),
        (1.0, 1.1, 0.20),
        (2.0, 0.9, 0.15),
        (2.0, 1.1, 0.25),
    ],
)
def test_grid_node_volatility_recovery(surface, maturity, strike, expected):
    assert surface.volatility(maturity, strike) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("maturity", "strike", "expected"),
    [
        (1.0, 0.9, 0.01),
        (1.0, 1.1, 0.04),
        (2.0, 0.9, 0.045),
        (2.0, 1.1, 0.125),
    ],
)
def test_grid_node_total_variance(surface, maturity, strike, expected):
    assert surface.total_variance(maturity, strike) == pytest.approx(expected)


def test_strike_interpolation_uses_total_variance(surface):
    expected = (0.01 + 0.04) / 2

    assert surface.total_variance(1.0, 1.0) == pytest.approx(expected)
    assert surface.volatility(1.0, 1.0) == pytest.approx(np.sqrt(expected))


def test_maturity_interpolation_uses_total_variance(surface):
    expected = (0.01 + 0.045) / 2

    assert surface.total_variance(1.5, 0.9) == pytest.approx(expected)
    assert surface.volatility(1.5, 0.9) == pytest.approx(np.sqrt(expected / 1.5))


def test_bilinear_total_variance_interpolation(surface):
    expected = (0.01 + 0.04 + 0.045 + 0.125) / 4

    assert surface.total_variance(1.5, 1.0) == pytest.approx(expected)
    assert surface.volatility(1.5, 1.0) == pytest.approx(np.sqrt(expected / 1.5))


def test_zero_volatility_is_supported():
    surface = FXVolSurface(
        maturities=(1.0, 2.0),
        strikes=(1.0,),
        volatilities=((0.0,), (0.0,)),
    )

    assert surface.total_variance(1.5, 1.0) == 0.0
    assert surface.volatility(1.5, 1.0) == 0.0


def test_single_grid_node_is_supported():
    surface = FXVolSurface(
        maturities=(1.0,),
        strikes=(1.1,),
        volatilities=((0.2,),),
    )

    assert surface.volatility(1.0, 1.1) == pytest.approx(0.2)


@pytest.mark.parametrize("maturity", [0.5, 2.5])
def test_maturity_extrapolation_rejected(surface, maturity):
    with pytest.raises(ValueError, match="maturity outside"):
        surface.volatility(maturity, 1.0)


@pytest.mark.parametrize("strike", [0.8, 1.2])
def test_strike_extrapolation_rejected(surface, strike):
    with pytest.raises(ValueError, match="strike outside"):
        surface.volatility(1.5, strike)


@pytest.mark.parametrize("maturity", [0.0, -1.0, np.nan, np.inf])
def test_invalid_query_maturity_rejected(surface, maturity):
    with pytest.raises(ValueError):
        surface.volatility(maturity, 1.0)


@pytest.mark.parametrize("strike", [0.0, -1.0, np.nan, np.inf])
def test_invalid_query_strike_rejected(surface, strike):
    with pytest.raises(ValueError):
        surface.volatility(1.0, strike)


@pytest.mark.parametrize(
    "maturities",
    [
        (),
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (1.0, np.nan),
        (1.0, np.inf),
    ],
)
def test_invalid_maturity_grid_rejected(maturities):
    with pytest.raises(ValueError):
        FXVolSurface(
            maturities=maturities,
            strikes=(1.0,),
            volatilities=tuple((0.2,) for _ in maturities),
        )


@pytest.mark.parametrize(
    "strikes",
    [
        (),
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (1.0, np.nan),
        (1.0, np.inf),
    ],
)
def test_invalid_strike_grid_rejected(strikes):
    with pytest.raises(ValueError):
        FXVolSurface(
            maturities=(1.0,),
            strikes=strikes,
            volatilities=(tuple(0.2 for _ in strikes),),
        )


@pytest.mark.parametrize(
    "volatilities",
    [
        ((0.2,),),
        ((0.2, 0.3),),
        ((0.2, 0.3), (0.4,)),
        ((-0.1, 0.2), (0.2, 0.3)),
        ((np.nan, 0.2), (0.2, 0.3)),
        ((np.inf, 0.2), (0.2, 0.3)),
    ],
)
def test_invalid_volatility_grid_rejected(volatilities):
    with pytest.raises(ValueError):
        FXVolSurface(
            maturities=(1.0, 2.0),
            strikes=(0.9, 1.1),
            volatilities=volatilities,
        )


def test_input_quotes_are_normalized_to_immutable_tuples():
    surface = FXVolSurface(
        maturities=[1, 2],
        strikes=[0.9, 1.1],
        volatilities=[[0.1, 0.2], [0.15, 0.25]],
    )

    assert isinstance(surface.maturities, tuple)
    assert isinstance(surface.strikes, tuple)
    assert isinstance(surface.volatilities, tuple)
    assert all(isinstance(row, tuple) for row in surface.volatilities)
