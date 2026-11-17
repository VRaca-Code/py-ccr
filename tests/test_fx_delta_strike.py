"""Regression tests for FX unadjusted forward-delta strike conversion."""

import math

import pytest
from scipy.stats import norm

from pyccr.market.fx_delta_strike import fx_forward_delta_to_strike


def recovered_forward_delta(
    forward: float,
    strike: float,
    maturity: float,
    volatility: float,
    option_type: str,
) -> float:
    """Independently recover signed forward delta from the strike."""
    sigma_sqrt_t = volatility * math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * volatility**2 * maturity) / sigma_sqrt_t

    call_delta = norm.cdf(d1)

    if option_type == "call":
        return call_delta

    return call_delta - 1.0


@pytest.mark.parametrize(
    ("forward", "maturity", "volatility", "delta", "option_type"),
    [
        (1.10, 1.0, 0.12, 0.25, "call"),
        (1.10, 1.0, 0.12, -0.25, "put"),
        (1.25, 0.5, 0.18, 0.10, "call"),
        (1.25, 0.5, 0.18, -0.10, "put"),
        (150.0, 2.0, 0.15, 0.40, "call"),
        (150.0, 2.0, 0.15, -0.40, "put"),
        (0.75, 5.0, 0.25, 0.50, "call"),
        (0.75, 5.0, 0.25, -0.50, "put"),
    ],
)
def test_strike_recovers_requested_forward_delta(
    forward,
    maturity,
    volatility,
    delta,
    option_type,
):
    strike = fx_forward_delta_to_strike(
        forward=forward,
        maturity=maturity,
        volatility=volatility,
        delta=delta,
        option_type=option_type,
    )

    recovered = recovered_forward_delta(
        forward,
        strike,
        maturity,
        volatility,
        option_type,
    )

    assert math.isfinite(strike)
    assert strike > 0
    assert recovered == pytest.approx(delta, abs=1e-12)


def test_25_delta_put_and_call_straddle_forward():
    forward = 1.10

    put_strike = fx_forward_delta_to_strike(
        forward=forward,
        maturity=1.0,
        volatility=0.12,
        delta=-0.25,
        option_type="put",
    )

    call_strike = fx_forward_delta_to_strike(
        forward=forward,
        maturity=1.0,
        volatility=0.12,
        delta=0.25,
        option_type="call",
    )

    assert put_strike < forward < call_strike


def test_forward_scaling():
    parameters = {
        "maturity": 1.5,
        "volatility": 0.20,
        "delta": 0.25,
        "option_type": "call",
    }

    strike_a = fx_forward_delta_to_strike(
        forward=1.0,
        **parameters,
    )

    strike_b = fx_forward_delta_to_strike(
        forward=2.0,
        **parameters,
    )

    assert strike_b == pytest.approx(2.0 * strike_a)


def test_call_strike_decreases_with_call_delta():
    parameters = {
        "forward": 1.10,
        "maturity": 1.0,
        "volatility": 0.15,
        "option_type": "call",
    }

    strike_10d = fx_forward_delta_to_strike(
        delta=0.10,
        **parameters,
    )

    strike_25d = fx_forward_delta_to_strike(
        delta=0.25,
        **parameters,
    )

    assert strike_10d > strike_25d


def test_put_strike_increases_with_absolute_put_delta():
    parameters = {
        "forward": 1.10,
        "maturity": 1.0,
        "volatility": 0.15,
        "option_type": "put",
    }

    strike_10d = fx_forward_delta_to_strike(
        delta=-0.10,
        **parameters,
    )

    strike_25d = fx_forward_delta_to_strike(
        delta=-0.25,
        **parameters,
    )

    assert strike_10d < strike_25d


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("forward", 0.0),
        ("forward", -1.0),
        ("maturity", 0.0),
        ("maturity", -0.5),
        ("volatility", 0.0),
        ("volatility", -0.1),
    ],
)
def test_nonpositive_market_inputs_rejected(field, value):
    parameters = {
        "forward": 1.10,
        "maturity": 1.0,
        "volatility": 0.12,
        "delta": 0.25,
        "option_type": "call",
    }
    parameters[field] = value

    with pytest.raises(ValueError, match="positive"):
        fx_forward_delta_to_strike(**parameters)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("forward", math.nan),
        ("forward", math.inf),
        ("maturity", math.nan),
        ("maturity", -math.inf),
        ("volatility", math.inf),
        ("volatility", math.nan),
        ("delta", math.nan),
        ("delta", math.inf),
    ],
)
def test_nonfinite_inputs_rejected(field, value):
    parameters = {
        "forward": 1.10,
        "maturity": 1.0,
        "volatility": 0.12,
        "delta": 0.25,
        "option_type": "call",
    }
    parameters[field] = value

    with pytest.raises(ValueError, match="finite"):
        fx_forward_delta_to_strike(**parameters)


@pytest.mark.parametrize(
    ("delta", "option_type"),
    [
        (0.0, "call"),
        (1.0, "call"),
        (-0.25, "call"),
        (1.25, "call"),
        (0.0, "put"),
        (-1.0, "put"),
        (0.25, "put"),
        (-1.25, "put"),
    ],
)
def test_invalid_signed_deltas_rejected(delta, option_type):
    with pytest.raises(ValueError, match="delta"):
        fx_forward_delta_to_strike(
            forward=1.10,
            maturity=1.0,
            volatility=0.12,
            delta=delta,
            option_type=option_type,
        )


def test_invalid_option_type_rejected():
    with pytest.raises(ValueError, match="option_type"):
        fx_forward_delta_to_strike(
            forward=1.10,
            maturity=1.0,
            volatility=0.12,
            delta=0.25,
            option_type="straddle",
        )


def test_delta_too_close_to_numerical_boundary_rejected():
    with pytest.raises(ValueError, match="boundary"):
        fx_forward_delta_to_strike(
            forward=1.10,
            maturity=1.0,
            volatility=0.12,
            delta=-1e-20,
            option_type="put",
        )


def test_extreme_volatility_time_product_rejected():
    with pytest.raises(ValueError, match="Volatility-time"):
        fx_forward_delta_to_strike(
            forward=1.10,
            maturity=1e308,
            volatility=1e308,
            delta=0.25,
            option_type="call",
        )


def test_strike_overflow_rejected():
    with pytest.raises(OverflowError, match="Strike"):
        fx_forward_delta_to_strike(
            forward=1.10,
            maturity=1.0,
            volatility=100.0,
            delta=0.25,
            option_type="call",
        )


def test_strike_underflow_rejected():
    with pytest.raises(OverflowError, match="Strike"):
        fx_forward_delta_to_strike(
            forward=1.10,
            maturity=1.0,
            volatility=100.0,
            delta=1.0 - 1e-12,
            option_type="call",
        )
