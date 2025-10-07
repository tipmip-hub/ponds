from ponds.core import PONDS
from ponds.shapegen import ShapeMethod, SINGLE, ELLIPSIS, CUSTOM, BLOB
from ponds.shiftgen import ShiftMethod, SIGMOID
from ponds.perturbationgen import PerturbationMethod, WhiteNoise, Trend
from ponds.correlationgen import CorrelationMethod, Noise, Gauss, Custom

__all__ = [
    "PONDS",
    "ShapeMethod",
    "SINGLE",
    "ELLIPSIS",
    "CUSTOM",
    "BLOB",
    "ShiftMethod",
    "SIGMOID",
    "PerturbationMethod",
    "WhiteNoise",
    "Trend",
    "CorrelationMethod",
    "Noise",
    "Gauss",
    "Custom",
]

type_dict = {
    "SINGLE": SINGLE,
    "ELLIPSIS": ELLIPSIS,
    "CUSTOM": CUSTOM,
    "BLOB": BLOB,
    "SIGMOID": SIGMOID,
    "WhiteNoise": WhiteNoise,
    "Trend": Trend,
    "Noise": Noise,
    "Gauss": Gauss,
    "Custom": Custom,
}
