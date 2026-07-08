"""Tests for aef_loader.index module."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aef_loader.constants import DataSource
from aef_loader.index import AEFIndex
from aef_loader.types import AEFTileInfo
from pyproj import Transformer


class TestAEFIndex:
    """Tests for AEFIndex class."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_download_requires_project(self, tmp_path):
        """Test that download requires a GCP project for GCS source."""
        index = AEFIndex(source=DataSource.GCS, cache_dir=tmp_path)

        with pytest.raises(ValueError, match="gcp_project is required"):
            await index.download()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_download_uses_cache(self, tmp_path):
        """Test that download uses cached file if exists."""
        # Use the correct cache filename for GCS source
        cache_file = tmp_path / "aef_index_gcs.parquet"
        cache_file.touch()  # Create empty file

        index = AEFIndex(source=DataSource.GCS, gcp_project="test", cache_dir=tmp_path)
        result = await index.download()

        assert result == cache_file

    @pytest.mark.unit
    def test_load_not_found_raises(self):
        """Test that load raises if file doesn't exist."""
        index = AEFIndex(cache_dir=Path("/nonexistent"))

        with pytest.raises(FileNotFoundError):
            index.load()

    @pytest.mark.unit
    def test_load_with_mock_gdf(self, tmp_path, mock_gdf):
        """Test loading with a mock GeoDataFrame."""
        # Save mock gdf with correct cache filename
        index_path = tmp_path / "aef_index_gcs.parquet"
        mock_gdf.to_parquet(index_path)

        index = AEFIndex(source=DataSource.GCS, cache_dir=tmp_path)
        gdf = index.load()

        assert len(gdf) == 3
        assert "path" in gdf.columns

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_with_bbox(self, tmp_path, mock_gdf):
        """Test querying with bounding box filter."""
        # Save mock gdf with correct cache filename
        index_path = tmp_path / "aef_index_gcs.parquet"
        mock_gdf.to_parquet(index_path)

        index = AEFIndex(source=DataSource.GCS, cache_dir=tmp_path)
        index.load()

        # Query with bbox that overlaps first two tiles
        tiles = await index.query(bbox=(-122.5, 37.5, -122.15, 37.7))

        assert len(tiles) == 2
        assert all(isinstance(t, AEFTileInfo) for t in tiles)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_with_projected_bbox_crs(self, tmp_path, mock_gdf):
        """Test querying with a bbox supplied in a projected CRS."""
        index_path = tmp_path / "aef_index_gcs.parquet"
        mock_gdf.to_parquet(index_path)

        index = AEFIndex(source=DataSource.GCS, cache_dir=tmp_path)
        index.load()

        wgs84_bbox = (-122.5, 37.5, -122.15, 37.7)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
        projected_bbox = transformer.transform_bounds(*wgs84_bbox, densify_pts=21)

        wgs84_tiles = await index.query(bbox=wgs84_bbox)
        projected_tiles = await index.query(bbox=projected_bbox, bbox_crs="EPSG:32610")

        assert [tile.id for tile in projected_tiles] == [tile.id for tile in wgs84_tiles]

    @pytest.mark.unit
    def test_projected_bbox_uses_densified_bounds(self):
        """Test projected bbox conversion covers bowed edges beyond corner-only bounds."""
        bbox = (300000, 6600000, 700000, 7600000)
        transformer = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
        x_coords = [bbox[0], bbox[0], bbox[2], bbox[2]]
        y_coords = [bbox[1], bbox[3], bbox[1], bbox[3]]
        corner_lons, corner_lats = transformer.transform(x_coords, y_coords)
        corner_north = max(corner_lats)

        densified = AEFIndex._bbox_to_wgs84(bbox, "EPSG:32631")

        assert densified[3] > corner_north

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_with_year_filter(self, tmp_path, mock_gdf):
        """Test querying with year filter."""
        index_path = tmp_path / "aef_index_gcs.parquet"
        mock_gdf.to_parquet(index_path)

        index = AEFIndex(source=DataSource.GCS, cache_dir=tmp_path)
        index.load()

        # Query for 2023 only
        tiles = await index.query(years=2023)

        assert len(tiles) == 2  # Two tiles from 2023

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_with_year_range(self, tmp_path, mock_gdf):
        """Test querying with year range."""
        index_path = tmp_path / "aef_index_gcs.parquet"
        mock_gdf.to_parquet(index_path)

        index = AEFIndex(source=DataSource.GCS, cache_dir=tmp_path)
        index.load()

        # Query for 2022-2023
        tiles = await index.query(years=(2022, 2023))

        assert len(tiles) == 3  # All tiles

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_with_limit(self, tmp_path, mock_gdf):
        """Test querying with limit."""
        index_path = tmp_path / "aef_index_gcs.parquet"
        mock_gdf.to_parquet(index_path)

        index = AEFIndex(source=DataSource.GCS, cache_dir=tmp_path)
        index.load()

        tiles = await index.query(limit=1)

        assert len(tiles) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_no_results(self, tmp_path, mock_gdf):
        """Test querying with no matching results."""
        index_path = tmp_path / "aef_index_gcs.parquet"
        mock_gdf.to_parquet(index_path)

        index = AEFIndex(source=DataSource.GCS, cache_dir=tmp_path)
        index.load()

        # Query with bbox that doesn't overlap any tiles
        tiles = await index.query(bbox=(0, 0, 1, 1))

        assert len(tiles) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_download_passes_project_to_gcs_store(self, tmp_path):
        """Test that download passes gcp_project as requester-pays header."""
        index = AEFIndex(
            source=DataSource.GCS, gcp_project="my-project", cache_dir=tmp_path
        )

        with (
            patch("aef_loader.index.GCSStore") as mock_gcs,
            patch("aef_loader.index.obs") as mock_obs,
        ):
            mock_result = MagicMock()
            mock_result.bytes_async = AsyncMock(return_value=b"fake-data")
            mock_obs.get_async = AsyncMock(return_value=mock_result)

            await index.download()

            mock_gcs.assert_called_once_with(
                bucket="alphaearth_foundations",
                client_options={
                    "default_headers": {"x-goog-user-project": "my-project"}
                },
            )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_creates_tile_info(self, tmp_path, mock_gdf):
        """Test that query creates proper AEFTileInfo objects."""
        index_path = tmp_path / "aef_index_gcs.parquet"
        mock_gdf.to_parquet(index_path)

        index = AEFIndex(source=DataSource.GCS, cache_dir=tmp_path)
        index.load()

        tiles = await index.query(limit=1)
        tile = tiles[0]

        assert tile.id == "1"
        assert tile.path.startswith("gs://")
        assert tile.year == 2022
        assert tile.crs_epsg == 32610
        assert tile.source == DataSource.GCS
