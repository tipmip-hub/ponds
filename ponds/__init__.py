from ponds.core import PONDS
from ponds.blobgen import BlobsMethod, SINGLE, GAUSS, CUSTOM
from ponds.shiftgen import ShiftsMethod, SIGMOID
from ponds.perturbationgen import PerturbationMethod, WhiteNoise, Trend

__all__ = [
    "PONDS",
    "BlobsMethod",
    "SINGLE",
    "GAUSS",
    "CUSTOM",
    "ShiftsMethod",
    "SIGMOID",
    "PerturbationMethod",
    "WhiteNoise",
    "Trend",
]

type_dict = {
    "SINGLE": SINGLE,
    "GAUSS": GAUSS,
    "CUSTOM": CUSTOM,
    "SIGMOID": SIGMOID,
    "WhiteNoise": WhiteNoise,
    "Trend": Trend,
}
