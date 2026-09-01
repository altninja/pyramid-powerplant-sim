from __future__ import annotations

import math
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field


class Vector3D(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def unit(self) -> Vector3D:
        mag = self.magnitude()
        if mag == 0.0:
            return Vector3D(x=0.0, y=0.0, z=0.0)
        return Vector3D(x=self.x / mag, y=self.y / mag, z=self.z / mag)

    def dot(self, other: Vector3D) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3D) -> Vector3D:
        return Vector3D(
            x=self.y * other.z - self.z * other.y,
            y=self.z * other.x - self.x * other.z,
            z=self.x * other.y - self.y * other.x,
        )

    def distance_to(self, other: Vector3D) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def __sub__(self, other: Vector3D) -> Vector3D:
        return Vector3D(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3D:
        return Vector3D(x=self.x * scalar, y=self.y * scalar, z=self.z * scalar)

    def __rmul__(self, scalar: float) -> Vector3D:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vector3D:
        return Vector3D(x=self.x / scalar, y=self.y / scalar, z=self.z / scalar)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class BoundingBox3D(BaseModel):
    min_point: Vector3D
    max_point: Vector3D

    def center(self) -> Vector3D:
        return Vector3D(
            x=(self.min_point.x + self.max_point.x) * 0.5,
            y=(self.min_point.y + self.max_point.y) * 0.5,
            z=(self.min_point.z + self.max_point.z) * 0.5,
        )

    def dimensions(self) -> Vector3D:
        return Vector3D(
            x=abs(self.max_point.x - self.min_point.x),
            y=abs(self.max_point.y - self.min_point.y),
            z=abs(self.max_point.z - self.min_point.z),
        )

    def volume(self) -> float:
        dims = self.dimensions()
        return dims.x * dims.y * dims.z

    def contains_point(self, point: Vector3D) -> bool:
        return (
            self.min_point.x <= point.x <= self.max_point.x
            and self.min_point.y <= point.y <= self.max_point.y
            and self.min_point.z <= point.z <= self.max_point.z
        )


class ShaftGeometry(BaseModel):
    name: str
    start_point: Vector3D
    angle_degrees: float
    length: float
    width: float
    height: float
    heading: str

    def get_unit_vector(self) -> Vector3D:
        rad = math.radians(self.angle_degrees)
        cos_val = math.cos(rad)
        sin_val = math.sin(rad)
        if self.heading.lower() == "north":
            return Vector3D(x=0.0, y=-cos_val, z=sin_val).unit()
        elif self.heading.lower() == "south":
            return Vector3D(x=0.0, y=cos_val, z=sin_val).unit()
        elif self.heading.lower() == "east":
            return Vector3D(x=cos_val, y=0.0, z=sin_val).unit()
        elif self.heading.lower() == "west":
            return Vector3D(x=-cos_val, y=0.0, z=sin_val).unit()
        return Vector3D(x=0.0, y=0.0, z=1.0)

    def get_end_point(self) -> Vector3D:
        u = self.get_unit_vector()
        return self.start_point + u * self.length

    def volume(self) -> float:
        return self.length * self.width * self.height


class PassageGeometry(BaseModel):
    name: str
    start_point: Vector3D
    end_point: Vector3D
    length: float
    incline_angle_degrees: float
    width: float
    height: float

    def get_unit_vector(self) -> Vector3D:
        diff = self.end_point - self.start_point
        return diff.unit()

    def volume(self) -> float:
        return self.length * self.width * self.height


class ChamberGeometry(BaseModel):
    name: str
    center: Vector3D
    floor_datum: float
    bounding_box: BoundingBox3D
    survey_volume: float

    def volume(self) -> float:
        return self.survey_volume


class GraniteCofferGeometry(BaseModel):
    external_length: float = 2.278
    external_width: float = 0.977
    external_height: float = 1.048
    internal_length: float = 1.977
    internal_width: float = 0.677
    internal_height: float = 0.872

    def external_volume(self) -> float:
        return self.external_length * self.external_width * self.external_height

    def internal_volume(self) -> float:
        return self.internal_length * self.internal_width * self.internal_height

    def solid_granite_volume(self) -> float:
        return self.external_volume() - self.internal_volume()

    def mass_kg(self, granite_density: float = 2650.0) -> float:
        return self.solid_granite_volume() * granite_density


class RelievingTiersGeometry(BaseModel):
    num_tiers: int = 5
    tier_names: Tuple[str, ...] = (
        "Davison's Chamber",
        "Wellington's Chamber",
        "Nelson's Chamber",
        "Lady Arbuthnot's Chamber",
        "Campbell's Chamber",
    )
    total_granite_beams: int = 43
    mean_beam_span: float = 6.50
    mean_beam_width: float = 1.20
    mean_beam_depth: float = 1.50
    mean_beam_mass_kg: float = 35000.0


class GrandGalleryGeometry(BaseModel):
    length_along_incline: float = 46.61
    slope_angle_degrees: float = 26.041666666666668
    vertical_height: float = 8.60
    width_base: float = 2.09
    width_roof: float = 1.05
    central_trench_width: float = 1.05
    side_ramps_width: float = 0.52
    num_slot_pairs: int = 28
    slot_spacing: float = 1.68
    slot_length: float = 0.54
    slot_width: float = 0.16
    slot_depth: float = 0.28
    cavity_volume: float = 550.0

    def get_slot_positions(
        self,
        start_point: Vector3D = Vector3D(x=0.0, y=-2.88, z=21.20),
    ) -> List[Tuple[Vector3D, Vector3D]]:
        rad = math.radians(self.slope_angle_degrees)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        slots: List[Tuple[Vector3D, Vector3D]] = []

        west_offset_x = -0.525 - 0.26
        east_offset_x = 0.525 + 0.26

        for i in range(self.num_slot_pairs):
            s = i * self.slot_spacing
            y_i = start_point.y + s * cos_a
            z_i = start_point.z + s * sin_a
            west_pos = Vector3D(x=west_offset_x, y=y_i, z=z_i)
            east_pos = Vector3D(x=east_offset_x, y=y_i, z=z_i)
            slots.append((west_pos, east_pos))
        return slots


class PyramidGeometry(BaseModel):
    mean_base_side: float = 230.364
    base_north: float = 230.253
    base_south: float = 230.454
    base_east: float = 230.391
    base_west: float = 230.357
    height: float = 146.580
    slope_angle_degrees: float = 51.84444444444445
    royal_cubit_meters: float = 0.52360

    subterranean_chamber: ChamberGeometry = Field(
        default_factory=lambda: ChamberGeometry(
            name="Subterranean Chamber",
            center=Vector3D(x=0.0, y=-27.4, z=-30.0),
            floor_datum=-30.00,
            bounding_box=BoundingBox3D(
                min_point=Vector3D(x=-7.035, y=-31.575, z=-30.00),
                max_point=Vector3D(x=7.035, y=-23.225, z=-26.48),
            ),
            survey_volume=280.0,
        )
    )

    subterranean_pit_depth: float = 3.20
    subterranean_blind_passage_length: float = 16.38
    subterranean_blind_passage_cross_section: Tuple[float, float] = (0.74, 0.74)

    queens_chamber: ChamberGeometry = Field(
        default_factory=lambda: ChamberGeometry(
            name="Queen's Chamber",
            center=Vector3D(x=0.0, y=0.5, z=21.20),
            floor_datum=21.20,
            bounding_box=BoundingBox3D(
                min_point=Vector3D(x=-2.875, y=-2.115, z=21.20),
                max_point=Vector3D(x=2.875, y=3.115, z=27.43),
            ),
            survey_volume=160.0,
        )
    )

    queens_niche_height: float = 4.67
    queens_niche_base_width: float = 1.57
    queens_niche_depth: float = 1.04

    queens_shaft_north: ShaftGeometry = Field(
        default_factory=lambda: ShaftGeometry(
            name="Queen's Northern Shaft",
            start_point=Vector3D(x=0.0, y=-2.115, z=22.0),
            angle_degrees=39.11666666666667,
            length=65.0,
            width=0.21,
            height=0.21,
            heading="north",
        )
    )

    queens_shaft_south: ShaftGeometry = Field(
        default_factory=lambda: ShaftGeometry(
            name="Queen's Southern Shaft",
            start_point=Vector3D(x=0.0, y=3.115, z=22.0),
            angle_degrees=39.60777777777778,
            length=63.60,
            width=0.21,
            height=0.21,
            heading="south",
        )
    )

    grand_gallery: GrandGalleryGeometry = Field(default_factory=GrandGalleryGeometry)

    antechamber: ChamberGeometry = Field(
        default_factory=lambda: ChamberGeometry(
            name="Antechamber",
            center=Vector3D(x=0.0, y=12.5, z=43.03),
            floor_datum=43.03,
            bounding_box=BoundingBox3D(
                min_point=Vector3D(x=-0.875, y=11.025, z=43.03),
                max_point=Vector3D(x=0.875, y=13.975, z=46.84),
            ),
            survey_volume=19.67,
        )
    )

    antechamber_granite_leaf_thickness: float = 0.41

    kings_chamber: ChamberGeometry = Field(
        default_factory=lambda: ChamberGeometry(
            name="King's Chamber",
            center=Vector3D(x=0.0, y=15.0, z=43.03),
            floor_datum=43.03,
            bounding_box=BoundingBox3D(
                min_point=Vector3D(x=-5.235, y=12.3825, z=43.03),
                max_point=Vector3D(x=5.235, y=17.6175, z=48.87),
            ),
            survey_volume=320.0,
        )
    )

    kings_relieving_tiers: RelievingTiersGeometry = Field(
        default_factory=RelievingTiersGeometry
    )

    kings_shaft_north: ShaftGeometry = Field(
        default_factory=lambda: ShaftGeometry(
            name="King's Northern Shaft",
            start_point=Vector3D(x=0.0, y=12.38, z=44.0),
            angle_degrees=32.46666666666667,
            length=71.0,
            width=0.22,
            height=0.22,
            heading="north",
        )
    )

    kings_shaft_south: ShaftGeometry = Field(
        default_factory=lambda: ShaftGeometry(
            name="King's Southern Shaft",
            start_point=Vector3D(x=0.0, y=17.62, z=44.0),
            angle_degrees=45.0,
            length=53.0,
            width=0.22,
            height=0.22,
            heading="south",
        )
    )

    coffer: GraniteCofferGeometry = Field(default_factory=GraniteCofferGeometry)

    descending_passage: PassageGeometry = Field(
        default_factory=lambda: PassageGeometry(
            name="Descending Passage",
            start_point=Vector3D(x=0.0, y=-56.5, z=17.0),
            end_point=Vector3D(x=0.0, y=-27.4, z=-30.0),
            length=105.23,
            incline_angle_degrees=26.523055555555554,
            width=1.05,
            height=1.20,
        )
    )

    ascending_passage: PassageGeometry = Field(
        default_factory=lambda: PassageGeometry(
            name="Ascending Passage",
            start_point=Vector3D(x=0.0, y=-38.2, z=0.0),
            end_point=Vector3D(x=0.0, y=-2.88, z=21.20),
            length=39.28,
            incline_angle_degrees=26.041666666666668,
            width=1.05,
            height=1.20,
        )
    )

    def total_solid_volume(self) -> float:
        return (1.0 / 3.0) * (self.mean_base_side**2) * self.height


_DEFAULT_GEOMETRY = PyramidGeometry()


def get_chamber_volume(chamber_name: str) -> float:
    name_norm = chamber_name.lower().strip()
    if "subterranean" in name_norm or "sub" in name_norm:
        return _DEFAULT_GEOMETRY.subterranean_chamber.volume()
    elif "queen" in name_norm or "qc" in name_norm:
        return _DEFAULT_GEOMETRY.queens_chamber.volume()
    elif "king" in name_norm or "kc" in name_norm:
        return _DEFAULT_GEOMETRY.kings_chamber.volume()
    elif "antechamber" in name_norm:
        return _DEFAULT_GEOMETRY.antechamber.volume()
    elif "gallery" in name_norm:
        return _DEFAULT_GEOMETRY.grand_gallery.cavity_volume
    elif "coffer" in name_norm:
        return _DEFAULT_GEOMETRY.coffer.internal_volume()
    raise ValueError(f"Unknown chamber name: {chamber_name}")


def get_shaft_unit_vector(shaft_name: str) -> Vector3D:
    name_norm = shaft_name.lower().strip()
    if "king" in name_norm or "kc" in name_norm:
        if "north" in name_norm:
            return _DEFAULT_GEOMETRY.kings_shaft_north.get_unit_vector()
        elif "south" in name_norm:
            return _DEFAULT_GEOMETRY.kings_shaft_south.get_unit_vector()
    elif "queen" in name_norm or "qc" in name_norm:
        if "north" in name_norm:
            return _DEFAULT_GEOMETRY.queens_shaft_north.get_unit_vector()
        elif "south" in name_norm:
            return _DEFAULT_GEOMETRY.queens_shaft_south.get_unit_vector()
    elif "descending" in name_norm:
        return _DEFAULT_GEOMETRY.descending_passage.get_unit_vector()
    elif "ascending" in name_norm:
        return _DEFAULT_GEOMETRY.ascending_passage.get_unit_vector()
    raise ValueError(f"Unknown shaft or passage name: {shaft_name}")


def get_all_nodes() -> Dict[str, Vector3D]:
    return {
        "base_center": Vector3D(x=0.0, y=0.0, z=0.0),
        "pyramid_apex": Vector3D(x=0.0, y=0.0, z=_DEFAULT_GEOMETRY.height),
        "descending_entrance": _DEFAULT_GEOMETRY.descending_passage.start_point,
        "descending_bottom": _DEFAULT_GEOMETRY.descending_passage.end_point,
        "ascending_junction": _DEFAULT_GEOMETRY.ascending_passage.start_point,
        "ascending_top": _DEFAULT_GEOMETRY.ascending_passage.end_point,
        "subterranean_chamber": _DEFAULT_GEOMETRY.subterranean_chamber.center,
        "subterranean_pit": Vector3D(
            x=0.0, y=-27.4, z=-30.0 - _DEFAULT_GEOMETRY.subterranean_pit_depth
        ),
        "subterranean_blind_passage": Vector3D(
            x=0.0,
            y=-27.4 - _DEFAULT_GEOMETRY.subterranean_blind_passage_length,
            z=-30.0,
        ),
        "queens_chamber": _DEFAULT_GEOMETRY.queens_chamber.center,
        "queens_niche": Vector3D(x=2.875, y=0.5, z=21.20),
        "queens_shaft_north_start": _DEFAULT_GEOMETRY.queens_shaft_north.start_point,
        "queens_shaft_north_end": _DEFAULT_GEOMETRY.queens_shaft_north.get_end_point(),
        "queens_shaft_south_start": _DEFAULT_GEOMETRY.queens_shaft_south.start_point,
        "queens_shaft_south_end": _DEFAULT_GEOMETRY.queens_shaft_south.get_end_point(),
        "grand_gallery_start": Vector3D(x=0.0, y=-2.88, z=21.20),
        "grand_gallery_end": Vector3D(x=0.0, y=39.0, z=43.03),
        "antechamber": _DEFAULT_GEOMETRY.antechamber.center,
        "antechamber_granite_leaf": Vector3D(x=0.0, y=11.5, z=44.0),
        "kings_chamber": _DEFAULT_GEOMETRY.kings_chamber.center,
        "kings_coffer": Vector3D(x=-2.5, y=15.0, z=43.03),
        "kings_relieving_tier1": Vector3D(x=0.0, y=15.0, z=49.5),
        "kings_relieving_tier2": Vector3D(x=0.0, y=15.0, z=51.5),
        "kings_relieving_tier3": Vector3D(x=0.0, y=15.0, z=53.5),
        "kings_relieving_tier4": Vector3D(x=0.0, y=15.0, z=55.5),
        "kings_relieving_tier5": Vector3D(x=0.0, y=15.0, z=57.5),
        "kings_shaft_north_start": _DEFAULT_GEOMETRY.kings_shaft_north.start_point,
        "kings_shaft_north_end": _DEFAULT_GEOMETRY.kings_shaft_north.get_end_point(),
        "kings_shaft_south_start": _DEFAULT_GEOMETRY.kings_shaft_south.start_point,
        "kings_shaft_south_end": _DEFAULT_GEOMETRY.kings_shaft_south.get_end_point(),
    }


def get_grand_gallery_slot_positions() -> List[Tuple[Vector3D, Vector3D]]:
    return _DEFAULT_GEOMETRY.grand_gallery.get_slot_positions()
