from ponds.core import PONDS
from ponds.shapegen import ShapesMethod, SINGLE, ELLIPSIS, CUSTOM, BLOB
from ponds.shiftgen import ShiftsMethod, SIGMOID
from ponds.perturbationgen import PerturbationMethod, WhiteNoise, Trend

__all__ = [
    "PONDS",
    "ShapesMethod",
    "SINGLE",
    "ELLIPSIS",
    "CUSTOM",
    "BLOB",
    "ShiftsMethod",
    "SIGMOID",
    "PerturbationMethod",
    "WhiteNoise",
    "Trend",
]

type_dict = {
    "SINGLE": SINGLE,
    "ELLIPSIS": ELLIPSIS,
    "CUSTOM": CUSTOM,
    "BLOB": BLOB,
    "SIGMOID": SIGMOID,
    "WhiteNoise": WhiteNoise,
    "Trend": Trend,
}
