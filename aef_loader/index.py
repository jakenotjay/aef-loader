"""
AEF Index management - download and filter the geoparquet index.

Uses obstore for efficient GCS and S3 access.
Supports both Google Cloud Storage (GCS) and Source Cooperative (S3) backends.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import obstore as obs
from aef_loader.constants import (
    GCS_BUCKET,
    GCS_INDEX_BLOB,
    SOURCE_COOP_BUCKET,
    SOURCE_COOP_INDEX_BLOB,
    SOURCE_COOP_PREFIX,
    SOURCE_COOP_REGION,
    DataSource,
)
from aef_loader.types import (
    AEFTileInfo,
    BoundingBox,
    DateRange,
)
from obstore.store import GCSStore, S3Store
from shapely.geometry import box

logger = logging.getLogger(__name__)


class AEFIndex:
    """
    Manages the AEF GeoParquet index for efficient spatial/temporal queries.

    The index contains metadata about all AEF tiles including their
    bounding boxes, paths, and optionally pre-fetched COG header metadata.

    Supports both GCS (Google Cloud Storage) and Source Cooperative (AWS S3) backends.

    Example:
        >>> # GCS (requires GCP project for requester-pays)
        >>> index = AEFIndex(source=DataSource.GCS, gcp_project="my-project")
        >>> await index.download()
        >>> tiles = await index.query(bbox=(-122.5, 37.5, -122.0, 38.0), years=(2020, 2023))

        >>> # Source Cooperative (public, no auth required)
        >>> index = AEFIndex(source=DataSource.SOURCE_COOP)
        >>> await index.download()
        >>> tiles = await index.query(bbox=(-122.5, 37.5, -122.0, 38.0), years=(2020, 2023))
    """

    def __init__(
        self,
        source: DataSource = DataSource.GCS,
        gcp_project: str | None = None,
        cache_dir: Path | None = None,
    ):
        """
        Initialize AEF index manager.

        Args:
            source: Data source (GCS or SOURCE_COOP)
            gcp_project: GCP project ID for requester-pays bucket access (GCS only)
            cache_dir: Directory for caching the index (default: /tmp)
        """
        self.source = source
        self.gcp_project = gcp_project
        self.cache_dir = cache_dir or Path("/tmp")
        self._gdf: gpd.GeoDataFrame | None = None
        self._index_path: Path | None = None

    @property
    def _cache_filename(self) -> str:
        """Get cache filename based on data source."""
        if self.source == DataSource.SOURCE_COOP:
            return "aef_index_source_coop.parquet"
        return "aef_index_gcs.parquet"

    @property
    def _bucket(self) -> str:
        """Get bucket name based on data source."""
        if self.source == DataSource.SOURCE_COOP:
            return SOURCE_COOP_BUCKET
        return GCS_BUCKET

    @property
    def _index_blob(self) -> str:
        """Get index blob path based on data source."""
        if self.source == DataSource.SOURCE_COOP:
            return SOURCE_COOP_INDEX_BLOB
        return GCS_INDEX_BLOB

    async def download(
        self,
        force: bool = False,
        local_path: Path | None = None,
    ) -> Path:
        """
        Download the AEF index from cloud storage using obstore.

        Args:
            force: Force re-download even if cached
            local_path: Custom path for the index file

        Returns:
            Path to the downloaded index file
        """
        if local_path is None:
            local_path = self.cache_dir / self._cache_filename

        if local_path.exists() and not force:
            logger.info(f"Using cached AEF index at {local_path}")
            self._index_path = local_path
            return local_path

        if self.source == DataSource.SOURCE_COOP:
            logger.info(
                f"Downloading AEF index from s3://{self._bucket}/{self._index_blob}"
            )
            store = S3Store(
                bucket=self._bucket,
                region=SOURCE_COOP_REGION,
                skip_signature=True,  # Public bucket, no auth needed
            )
        else:
            # GCS - requires project for requester-pays
            if not self.gcp_project:
                raise ValueError(
                    "gcp_project is required for downloading from GCS requester-pays bucket"
                )

            logger.info(
                f"Downloading AEF index from gs://{self._bucket}/{self._index_blob}"
            )
            store = GCSStore(
                bucket=self._bucket,
                client_options={
                    "default_headers": {"x-goog-user-project": self.gcp_project}
                },
            )

        local_path.parent.mkdir(parents=True, exist_ok=True)

        result = await obs.get_async(store, self._index_blob)
        data = await result.bytes_async()

        local_path.write_bytes(data)
        logger.info(f"Downloaded AEF index to {local_path}")

        self._index_path = local_path
        return local_path

    def load(self, path: Path | None = None) -> gpd.GeoDataFrame:
        """
        Load the index into memory as a GeoDataFrame.

        Args:
            path: Path to index file (uses cached path if not provided)

        Returns:
            GeoDataFrame with AEF tile metadata
        """
        if path is None:
            path = self._index_path
        if path is None:
            path = self.cache_dir / self._cache_filename

        if not path.exists():
            raise FileNotFoundError(
                f"Index not found at {path}. Call download() first."
            )

        logger.info(f"Loading AEF index from {path}")
        self._gdf = gpd.read_parquet(path)
        logger.info(f"Loaded {len(self._gdf)} tiles from AEF index")
        return self._gdf

    def _convert_path_to_source(self, path: str) -> str:
        """
        Convert a path from index to the appropriate cloud path for the data source.

        The index stores GCS paths (gs://...), but Source Coop uses S3 paths.
        GCS: gs://alphaearth_foundations/satellite_embedding/v1/annual/YYYY/zone/file.tiff
        S3:  s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/YYYY/zone/file.tiff
        """
        if self.source == DataSource.SOURCE_COOP:
            # Convert GCS path to S3 path for Source Coop
            if path.startswith("gs://alphaearth_foundations/satellite_embedding/"):
                suffix = path.replace(
                    "gs://alphaearth_foundations/satellite_embedding/", ""
                )
                return f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_PREFIX}/{suffix}"
            elif path.startswith("gs://"):
                # Handle other GCS paths - try to extract the relative portion
                parts = path.split("/")
                # Find 'v1/annual' in the path and extract from there
                for i, part in enumerate(parts):
                    if part == "v1" and i + 1 < len(parts) and parts[i + 1] == "annual":
                        suffix = "/".join(parts[i:])
                        return (
                            f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_PREFIX}/{suffix}"
                        )
        return path

    def _get_start_and_end_year(self, years: int | DateRange) -> tuple[int, int]:
        if isinstance(years, int):
            start_year = end_year = years
            return start_year, end_year

        start_year, end_year = years

        # Handle string dates
        if isinstance(start_year, str):
            start_year = int(start_year[:4])
        if isinstance(end_year, str):
            end_year = int(end_year[:4])

        return start_year, end_year

    async def query(
        self,
        bbox: BoundingBox | None = None,
        years: int | DateRange | None = None,
        limit: int | None = None,
    ) -> list[AEFTileInfo]:
        """
        Query the index for tiles matching the given criteria.

        Args:
            bbox: Bounding box filter (minx, miny, maxx, maxy) in WGS84
            years: Single year or (start_year, end_year) tuple
            limit: Maximum number of tiles to return

        Returns:
            List of AEFTileInfo objects matching the query
        """
        if self._gdf is None:
            self.load()

        assert self._gdf is not None, "Index not loaded"
        gdf = self._gdf.copy()

        # Apply spatial filter
        if bbox:
            minx, miny, maxx, maxy = bbox
            bbox_geom = box(minx, miny, maxx, maxy)
            gdf = gdf[gdf.geometry.intersects(bbox_geom)]
            logger.info(f"After bbox filter: {len(gdf)} tiles")

        # Apply temporal filter
        if years is not None:
            start_year, end_year = self._get_start_and_end_year(years)
            gdf = gdf[(gdf["year"] >= start_year) & (gdf["year"] <= end_year)]
            logger.info(f"After year filter: {len(gdf)} tiles")

        if limit:
            gdf = gdf.head(limit)

        if len(gdf) == 0:
            return []

        tiles = []
        for _, row in gdf.iterrows():
            # Convert path to the appropriate cloud path for this source
            path = self._convert_path_to_source(row["path"])

            tile = AEFTileInfo(
                id=str(row.get("fid", row.name)),
                path=path,
                year=row["year"],
                bbox=(
                    row["wgs84_west"],
                    row["wgs84_south"],
                    row["wgs84_east"],
                    row["wgs84_north"],
                ),
                crs_epsg=int(row["crs"].split(":")[1])
                if ":" in str(row["crs"])
                else 4326,
                utm_zone=row.get("utm_zone"),
                utm_bounds=(
                    row["utm_west"],
                    row["utm_south"],
                    row["utm_east"],
                    row["utm_north"],
                ),
                source=self.source,
            )
            tiles.append(tile)

        return tiles

