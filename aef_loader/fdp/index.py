"""Forest Data Partnership index — built locally by listing GCS.

Unlike the AEF index (which is published as a GeoParquet), the FDP project
distributes only the COGs themselves. ``FDPIndex.build()`` discovers the
release/commodity/year hierarchy via prefix listings and writes a single
GeoParquet to the local cache, after which queries are served entirely from
the cached file — same shape as ``AEFIndex.download() → load() → query()``.

Tile bboxes are derived deterministically from the filename. Each tile is a
1°×1° EPSG:4326 COG named ``lng_{X}_lat_{Y}.tif`` where ``(X, Y)`` is the SW
corner — verified empirically against the GeoTIFF tags and rasterio bounds;
the public FDP docs incorrectly describe this as the NW corner.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import obstore as obs
import pandas as pd
from shapely.geometry import box

from aef_loader._cloud import default_cache_dir, make_gcs_store
from aef_loader.constants import (
    FDP_BUCKET,
    FDP_DEFAULT_RELEASE,
    FDP_PREFIX,
    FDP_TILE_SIZE_DEG,
    DataSource,
)
from aef_loader.types import (
    COMMODITIES,
    BoundingBox,
    Commodity,
    DateRange,
    FDPTileInfo,
)

logger = logging.getLogger(__name__)

# lng_{X}_lat_{Y}.tif where X, Y are signed integer degrees.
_TILE_FILENAME_RE = re.compile(r"^lng_(-?\d+)_lat_(-?\d+)\.tif$")


class FDPIndex:
    """Local GeoParquet index for FDP commodity-probability tiles.

    The first call to :meth:`build` lists the requester-pays bucket for the
    chosen release and writes ``fdp_index_{release}.parquet`` to the cache
    directory. Subsequent calls return the cached path unless ``force=True``.

    Example:
        ```python
        index = FDPIndex(release="2025b", gcp_project="my-project")
        await index.build()
        index.load()
        tiles = await index.query(
            bbox=(9.0, 5.0, 11.0, 7.0),
            years=2024,
            commodities=["coffee", "cocoa"],
        )
        ```
    """

    def __init__(
        self,
        release: str = FDP_DEFAULT_RELEASE,
        gcp_project: str | None = None,
        cache_dir: Path | None = None,
    ):
        """Initialise the FDP index.

        Args:
            release: Release name (subdirectory under ``forestdatapartnership/``,
                e.g. ``"2025b"``). One parquet is built per release.
            gcp_project: GCP project to bill for requester-pays egress.
            cache_dir: Directory for the cached parquet (defaults to
                :func:`aef_loader._cloud.default_cache_dir`).
        """
        self.release = release
        self.gcp_project = gcp_project
        self.cache_dir = cache_dir or default_cache_dir()
        self._gdf: gpd.GeoDataFrame | None = None
        self._index_path: Path | None = None

    @property
    def _cache_filename(self) -> str:
        return f"fdp_index_{self.release}.parquet"

    @property
    def source(self) -> DataSource:
        """Backing data source — FDP is GCS-only at present."""
        return DataSource.GCS

    async def build(
        self,
        force: bool = False,
        local_path: Path | None = None,
    ) -> Path:
        """Build (or reuse) the local GeoParquet index for this release.

        Args:
            force: When ``True``, rebuild even if a cached parquet exists.
            local_path: Override for the output path.

        Returns:
            Path to the GeoParquet index file.
        """
        if local_path is None:
            local_path = self.cache_dir / self._cache_filename

        if local_path.exists() and not force:
            logger.info(f"Using cached FDP index at {local_path}")
            self._index_path = local_path
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        store = make_gcs_store(FDP_BUCKET, self.gcp_project)
        release_prefix = f"{FDP_PREFIX}/{self.release}/"
        logger.info(f"Building FDP index from gs://{FDP_BUCKET}/{release_prefix}")

        commodities = await self._discover_commodities(store, release_prefix)
        if not commodities:
            raise ValueError(
                f"No known commodities found under gs://{FDP_BUCKET}/{release_prefix}"
            )
        logger.info(f"Discovered commodities: {commodities}")

        # Discover years per commodity in parallel.
        commodity_years = await asyncio.gather(
            *[self._discover_years(store, release_prefix, c) for c in commodities]
        )
        leaf_specs: list[tuple[Commodity, int]] = [
            (c, y) for c, ys in zip(commodities, commodity_years) for y in ys
        ]
        logger.info(f"Listing {len(leaf_specs)} (commodity, year) leaves: {leaf_specs}")

        # List every leaf concurrently and collect rows.
        per_leaf_rows = await asyncio.gather(
            *[
                self._list_leaf_tiles(store, release_prefix, commodity, year)
                for commodity, year in leaf_specs
            ]
        )
        rows = [row for leaf in per_leaf_rows for row in leaf]
        if not rows:
            raise ValueError(f"No tiles found under gs://{FDP_BUCKET}/{release_prefix}")

        gdf = self._rows_to_gdf(rows)
        gdf.to_parquet(local_path, index=False)
        logger.info(f"Wrote {len(gdf)} tiles to {local_path}")

        self._index_path = local_path
        return local_path

    async def _discover_commodities(
        self, store, release_prefix: str
    ) -> list[Commodity]:
        """List ``release/`` and intersect with the known commodity slugs.

        Excludes anything outside :data:`aef_loader.types.COMMODITIES` (notably
        ``palm_2023`` and any future non-standard layouts).
        """
        result = await obs.list_with_delimiter_async(store, prefix=release_prefix)
        found = {p.rsplit("/", 1)[-1] for p in result["common_prefixes"]}
        unknown = found - set(COMMODITIES)
        if unknown:
            logger.info(
                f"Skipping non-standard commodity directories: {sorted(unknown)}"
            )
        return [c for c in COMMODITIES if c in found]

    async def _discover_years(
        self, store, release_prefix: str, commodity: Commodity
    ) -> list[int]:
        """List the year subdirectories under a commodity prefix."""
        prefix = f"{release_prefix}{commodity}/"
        result = await obs.list_with_delimiter_async(store, prefix=prefix)
        years: list[int] = []
        for p in result["common_prefixes"]:
            leaf = p.rsplit("/", 1)[-1]
            if leaf.isdigit():
                years.append(int(leaf))
            else:
                logger.debug(f"Skipping non-year subdir: {p}")
        return sorted(years)

    async def _list_leaf_tiles(
        self,
        store,
        release_prefix: str,
        commodity: Commodity,
        year: int,
    ) -> list[dict]:
        """List every COG in one ``commodity/year/`` directory and parse rows."""
        prefix = f"{release_prefix}{commodity}/{year}/"
        rows: list[dict] = []
        skipped = 0
        async for batch in obs.list(store, prefix=prefix):
            for obj in batch:
                key = obj["path"]
                fname = key.rsplit("/", 1)[-1]
                m = _TILE_FILENAME_RE.match(fname)
                if not m:
                    if obj["size"] > 0:
                        skipped += 1
                    continue
                lng, lat = int(m.group(1)), int(m.group(2))
                rows.append(
                    {
                        "id": f"{commodity}_{year}_{fname[:-4]}",
                        "release": self.release,
                        "commodity": commodity,
                        "year": year,
                        "lng": lng,
                        "lat": lat,
                        "path": f"gs://{FDP_BUCKET}/{key}",
                        "geometry": box(
                            lng,
                            lat,
                            lng + FDP_TILE_SIZE_DEG,
                            lat + FDP_TILE_SIZE_DEG,
                        ),
                    }
                )
        if skipped:
            logger.warning(
                f"{prefix}: skipped {skipped} files not matching lng_*_lat_*.tif"
            )
        logger.info(f"{prefix}: {len(rows)} tiles")
        return rows

    @staticmethod
    def _rows_to_gdf(rows: list[dict]) -> gpd.GeoDataFrame:
        df = pd.DataFrame(rows)
        return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    def load(self, path: Path | None = None) -> gpd.GeoDataFrame:
        """Load the cached GeoParquet into memory."""
        if path is None:
            path = self._index_path or (self.cache_dir / self._cache_filename)

        if not path.exists():
            raise FileNotFoundError(
                f"FDP index not found at {path}. Call build() first."
            )

        logger.info(f"Loading FDP index from {path}")
        self._gdf = gpd.read_parquet(path)
        logger.info(f"Loaded {len(self._gdf)} tiles from FDP index")
        return self._gdf

    @staticmethod
    def _year_range(years: int | DateRange) -> tuple[int, int]:
        if isinstance(years, int):
            return years, years
        start, end = years
        if isinstance(start, str):
            start = int(start[:4])
        if isinstance(end, str):
            end = int(end[:4])
        return start, end

    async def query(
        self,
        bbox: BoundingBox | None = None,
        years: int | DateRange | None = None,
        commodities: Iterable[Commodity] | None = None,
        limit: int | None = None,
    ) -> list[FDPTileInfo]:
        """Query the cached index for tiles matching the given filters.

        Args:
            bbox: ``(minx, miny, maxx, maxy)`` in WGS84.
            years: Single year or ``(start, end)`` inclusive.
            commodities: Restrict to a subset of commodities.
            limit: Maximum number of tiles to return.

        Returns:
            List of :class:`FDPTileInfo`.
        """
        if self._gdf is None:
            self.load()
        assert self._gdf is not None
        gdf = self._gdf

        if commodities is not None:
            commodities = list(commodities)
            gdf = gdf[gdf["commodity"].isin(commodities)]
            logger.info(f"After commodity filter: {len(gdf)} tiles")

        if bbox is not None:
            minx, miny, maxx, maxy = bbox
            gdf = gdf[gdf.geometry.intersects(box(minx, miny, maxx, maxy))]
            logger.info(f"After bbox filter: {len(gdf)} tiles")

        if years is not None:
            start, end = self._year_range(years)
            gdf = gdf[(gdf["year"] >= start) & (gdf["year"] <= end)]
            logger.info(f"After year filter: {len(gdf)} tiles")

        if limit:
            gdf = gdf.head(limit)

        if len(gdf) == 0:
            return []

        tiles: list[FDPTileInfo] = []
        for _, row in gdf.iterrows():
            lng, lat = int(row["lng"]), int(row["lat"])
            tiles.append(
                FDPTileInfo(
                    id=str(row["id"]),
                    path=str(row["path"]),
                    year=int(row["year"]),
                    bbox=(
                        float(lng),
                        float(lat),
                        float(lng + FDP_TILE_SIZE_DEG),
                        float(lat + FDP_TILE_SIZE_DEG),
                    ),
                    commodity=row["commodity"],
                    release=str(row["release"]),
                )
            )
        return tiles
