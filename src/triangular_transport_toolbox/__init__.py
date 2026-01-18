"""
Triangular Transport Toolbox

A Python package for triangular transport maps in probabilistic modeling.
"""

from triangular_transport_toolbox.rectifier import Rectifier
from triangular_transport_toolbox.transport_map import transport_map

__version__ = "1.0.0"
__all__ = ["transport_map", "Rectifier"]
