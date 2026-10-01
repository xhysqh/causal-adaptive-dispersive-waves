from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class MCHWeakFormContract:
    """Frozen weak-solution semantics for the cubic/FORQ mCH equation."""

    equation: str = "cubic_forq_mch"
    boundary: str = "periodic"
    sector: str = "conservative_lax"
    peak_product: str = "arithmetic_average_of_squared_one_sided_derivatives"
    collision_policy: str = "stop_before_collision"
    green_normalization: str = "unit_peak_periodic"

    def __post_init__(self) -> None:
        expected = {
            "equation": "cubic_forq_mch",
            "boundary": "periodic",
            "sector": "conservative_lax",
            "peak_product": (
                "arithmetic_average_of_squared_one_sided_derivatives"
            ),
            "collision_policy": "stop_before_collision",
            "green_normalization": "unit_peak_periodic",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"Unsupported mCH weak-form choice {name}={getattr(self, name)!r}; "
                    f"M0 permits only {value!r}."
                )

    def to_metadata(self) -> dict[str, str]:
        return asdict(self)
