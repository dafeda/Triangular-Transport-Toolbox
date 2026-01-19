"""
Triangular Transport Toolbox

A Python package for triangular transport maps in probabilistic modeling.
"""

from triangular_transport_toolbox.monotonicity import (
    IntegratedRectifier,
    MonotonicityStrategy,
    SeparableMonotonicity,
)
from triangular_transport_toolbox.rectifier import Rectifier
from triangular_transport_toolbox.transport_map import TransportMap

__version__ = "1.0.0"
__all__ = [
    "TransportMap",
    "Rectifier",
    "MonotonicityStrategy",
    "IntegratedRectifier",
    "SeparableMonotonicity",
]
