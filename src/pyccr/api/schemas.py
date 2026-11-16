"""Validated request and response models for the PyCCR API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BachelierPriceRequest(BaseModel):
    """Inputs for European swaption pricing under the Bachelier model."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    forward: float = Field(description="Forward swap rate in absolute units.")
    strike: float = Field(description="Strike rate in absolute units.")
    expiry: float = Field(ge=0, description="Option expiry in years.")
    normal_volatility: float = Field(
        ge=0, description="Normal volatility in absolute units."
    )
    annuity: float = Field(gt=0, description="Swap annuity.")
    option_type: Literal["payer", "receiver"] = "payer"
    notional: float = Field(gt=0, default=1.0)


class BachelierPriceResponse(BaseModel):
    """Bachelier swaption pricing result."""

    price: float
    currency: str = "unspecified"
    model: str = "Bachelier"
