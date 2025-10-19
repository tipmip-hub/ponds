from ponds.core import PONDS
from ponds.shapegen import ShapeMethod, SINGLE, ELLIPSIS, CUSTOM, BLOB
from ponds.shiftgen import ShiftMethod, SIGMOID
from ponds.backgroundgen import BackgroundMethod, WhiteNoise, Trend
from ponds.correlationgen import CorrelationMethod, NoCorr, Noise, Gauss, Custom
from typing import Any, Dict

__all__ = [
    "PONDS",
    "ShapeMethod",
    "SINGLE",
    "ELLIPSIS",
    "CUSTOM",
    "BLOB",
    "ShiftMethod",
    "SIGMOID",
    "BackgroundMethod",
    "WhiteNoise",
    "Trend",
    "CorrelationMethod",
    "NoCorr",
    "Noise",
    "Gauss",
    "Custom",
]
type_dict: Dict[str, Any] = {
    "SINGLE": SINGLE,
    "ELLIPSIS": ELLIPSIS,
    "CUSTOM": CUSTOM,
    "BLOB": BLOB,
    "SIGMOID": SIGMOID,
    "WhiteNoise": WhiteNoise,
    "Trend": Trend,
    "NoCorr": NoCorr,
    "Noise": Noise,
    "Gauss": Gauss,
    "Custom": Custom,
}
