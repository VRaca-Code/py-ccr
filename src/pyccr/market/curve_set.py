"""Multi-curve market-data container."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pyccr.market.discount_curve import DiscountCurve


@dataclass(frozen=True)
class CurveSet:
    """Validated discount and forward-projection curves."""

    discount_curves: Mapping[str, DiscountCurve]
    projection_curves: Mapping[tuple[str, str], DiscountCurve]

    def __post_init__(self) -> None:
        discount = {}
        projection = {}

        for currency, curve in self.discount_curves.items():
            if not isinstance(currency, str) or not currency.strip():
                raise ValueError("currency must be a non-empty string")

            if not isinstance(curve, DiscountCurve):
                raise TypeError("discount curve must be a DiscountCurve")

            key = currency.strip().upper()

            if key in discount:
                raise ValueError("duplicate discount curve currency")

            discount[key] = curve

        for key, curve in self.projection_curves.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not all(isinstance(x, str) and x.strip() for x in key)
            ):
                raise ValueError("projection keys must be (currency, index) tuples")

            if not isinstance(curve, DiscountCurve):
                raise TypeError("projection curve must be a DiscountCurve")

            currency, index = (x.strip().upper() for x in key)
            normalized_key = (currency, index)

            if normalized_key in projection:
                raise ValueError("duplicate projection curve key")

            projection[normalized_key] = curve

        object.__setattr__(
            self,
            "discount_curves",
            MappingProxyType(discount),
        )
        object.__setattr__(
            self,
            "projection_curves",
            MappingProxyType(projection),
        )

    def discount_curve(self, currency: str) -> DiscountCurve:
        """Return the discount curve for a currency."""
        if not isinstance(currency, str) or not currency.strip():
            raise ValueError("currency must be a non-empty string")

        key = currency.strip().upper()

        try:
            return self.discount_curves[key]
        except KeyError as exc:
            raise KeyError(f"No discount curve configured for {key}") from exc

    def projection_curve(
        self,
        currency: str,
        index: str,
    ) -> DiscountCurve:
        """Return the projection curve for an interest-rate index."""
        if (
            not isinstance(currency, str)
            or not currency.strip()
            or not isinstance(index, str)
            or not index.strip()
        ):
            raise ValueError("currency and index must be non-empty strings")

        key = (currency.strip().upper(), index.strip().upper())

        try:
            return self.projection_curves[key]
        except KeyError as exc:
            raise KeyError(f"No projection curve configured for {key}") from exc
