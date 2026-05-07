"""Type definitions for AEF loader."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

from aef_loader.constants import DataSource

# Type aliases
BoundingBox = tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
DateRange = tuple[
    str | int, str | int
]  # (start, end) as YYYY-MM-DD strings or year integers

# Forest Data Partnership commodity slugs. Restricted to the four production
# commodity-probability layers; ``palm_2023`` is intentionally excluded as it
# uses a non-standard layout (flat, no year subdir).
Commodity = Literal["cocoa", "coffee", "palm", "rubber"]
COMMODITIES: tuple[Commodity, ...] = ("cocoa", "coffee", "palm", "rubber")


@dataclass
class AEFTileInfo:
    """Information about an AEF tile/scene."""

    id: str
    path: str  # Cloud path (gs://bucket/key or s3://bucket/key)
    year: int

    # Bounding box in WGS84
    bbox: BoundingBox

    # Native CRS
    crs_epsg: int
    utm_zone: str | None = None

    # UTM bounds (west, south, east, north) in native CRS
    utm_bounds: BoundingBox | None = None

    # Data source (GCS or Source Cooperative)
    source: DataSource | None = None

    @property
    def as_datetime(self) -> dt.datetime:
        """Get datetime (January 1st of the year)."""
        return dt.datetime(self.year, 1, 1)


@dataclass
class FDPTileInfo:
    """Information about a Forest Data Partnership commodity-probability tile.

    Each tile is a 1°×1° EPSG:4326 float32 COG. The tile filename encodes the
    SW corner coordinates as integers; ``bbox`` is therefore exactly
    ``(lng, lat, lng + 1, lat + 1)``.
    """

    id: str
    path: str  # gs://bucket/key
    year: int
    bbox: BoundingBox
    commodity: Commodity
    release: str

    # Constant for FDP — included so FDPTileInfo and AEFTileInfo share a
    # minimum schema (id, path, year, bbox, crs_epsg) that future helpers can
    # accept structurally.
    crs_epsg: int = 4326
    source: DataSource = DataSource.GCS

    @property
    def lng(self) -> int:
        """West edge longitude (integer degrees)."""
        return int(self.bbox[0])

    @property
    def lat(self) -> int:
        """South edge latitude (integer degrees)."""
        return int(self.bbox[1])

    @property
    def as_datetime(self) -> dt.datetime:
        """Get datetime (January 1st of the year)."""
        return dt.datetime(self.year, 1, 1)
