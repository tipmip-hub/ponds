from ponds.core import PONDS
from ponds.shapegen import ShapesMethod, SINGLE, BLOB, CUSTOM
from ponds.shiftgen import ShiftsMethod, SIGMOID
from ponds.perturbationgen import PerturbationMethod, WhiteNoise, Trend

__all__ = [
    "PONDS",
    "ShapesMethod",
    "SINGLE",
    "BLOB",
    "CUSTOM",
    "ShiftsMethod",
    "SIGMOID",
    "PerturbationMethod",
    "WhiteNoise",
    "Trend",
]

type_dict = {
    "SINGLE": SINGLE,
    "BLOB": BLOB,
    "CUSTOM": CUSTOM,
    "SIGMOID": SIGMOID,
    "WhiteNoise": WhiteNoise,
    "Trend": Trend,
}
