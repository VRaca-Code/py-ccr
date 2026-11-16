"""Swaption pricing endpoints."""

from fastapi import APIRouter

from pyccr.api.schemas import BachelierPriceRequest, BachelierPriceResponse
from pyccr.pricing.bachelier_swaption import bachelier_swaption_price

router = APIRouter(prefix="/api/v1/swaptions", tags=["Swaptions"])


@router.post(
    "/bachelier/price",
    response_model=BachelierPriceResponse,
)
def price_bachelier(
    request: BachelierPriceRequest,
) -> BachelierPriceResponse:
    """Price a European payer or receiver swaption using normal volatility."""
    price = bachelier_swaption_price(
        forward=request.forward,
        strike=request.strike,
        expiry=request.expiry,
        normal_volatility=request.normal_volatility,
        annuity=request.annuity,
        option_type=request.option_type,
        notional=request.notional,
    )

    return BachelierPriceResponse(price=price)
