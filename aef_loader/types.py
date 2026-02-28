"""Type definitions for AEF loader."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from aef_loader.constants import DataSource

# Type aliases
BoundingBox = tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
DateRange = tuple[
    str | int, str | int
]  # (start, end) as YYYY-MM-DD strings or year integers


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
