"""Pyramid Power Plant Multi-Physics Simulation Engine.

A mathematically rigorous simulation engine modeling Christopher Dunn's
Giza Power Plant hypothesis with coupled acoustic, fluid, chemical,
electromechanical, and electromagnetic physical subsystems.
"""

from engine.config import (
    ROYAL_CUBIT_TO_METERS,
    GraniteProperties,
    LimestoneProperties,
    HydrogenReactionProperties,
    GasProperties,
    MaserProperties,
    SchumannProperties,
    HydraulicProperties,
    SimulationConfig,
)
from engine.geometry import (
    Vector3D,
    BoundingBox3D,
    ChamberGeometry,
    PassageGeometry,
    ShaftGeometry,
    PyramidGeometry,
    get_chamber_volume,
    get_shaft_unit_vector,
    get_all_nodes,
    get_grand_gallery_slot_positions,
)

__all__ = [
    "ROYAL_CUBIT_TO_METERS",
    "GraniteProperties",
    "LimestoneProperties",
    "HydrogenReactionProperties",
    "GasProperties",
    "MaserProperties",
    "SchumannProperties",
    "HydraulicProperties",
    "SimulationConfig",
    "Vector3D",
    "BoundingBox3D",
    "ChamberGeometry",
    "PassageGeometry",
    "ShaftGeometry",
    "PyramidGeometry",
    "get_chamber_volume",
    "get_shaft_unit_vector",
    "get_all_nodes",
    "get_grand_gallery_slot_positions",
]
