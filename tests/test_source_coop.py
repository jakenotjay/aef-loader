"""Tests for Source Cooperative (S3) support in aef_loader."""

import pytest
from aef_loader.constants import (
    GCS_BUCKET,
    SOURCE_COOP_BUCKET,
    DataSource,
)
from aef_loader.index import AEFIndex
from aef_loader.reader import (
    _detect_protocol,
    _parse_cloud_path,
    _parse_s3_path,
)
from aef_loader.types import AEFTileInfo


class TestPathParsing:
    """Tests for cloud path parsing functions."""

    @pytest.mark.unit
    def test_parse_s3_path_with_prefix(self):
        """Test parsing S3 path with s3:// prefix."""
        bucket, key = _parse_s3_path("s3://my-bucket/path/to/file.tif")

        assert bucket == "my-bucket"
        assert key == "path/to/file.tif"

    @pytest.mark.unit
    def test_parse_s3_path_without_prefix(self):
        """Test parsing S3 path without s3:// prefix."""
        bucket, key = _parse_s3_path("my-bucket/path/to/file.tif")

        assert bucket == "my-bucket"
        assert key == "path/to/file.tif"

    @pytest.mark.unit
    def test_parse_s3_path_bucket_only(self):
        """Test parsing bucket-only S3 path."""
        bucket, key = _parse_s3_path("s3://my-bucket")

        assert bucket == "my-bucket"
        assert key == ""

    @pytest.mark.unit
    def test_detect_protocol_gs(self):
        """Test detecting GCS protocol."""
        assert _detect_protocol("gs://bucket/key") == "gs"

    @pytest.mark.unit
    def test_detect_protocol_s3(self):
        """Test detecting S3 protocol."""
        assert _detect_protocol("s3://bucket/key") == "s3"

    @pytest.mark.unit
    def test_detect_protocol_unknown(self):
        """Test detecting unknown protocol raises error."""
        with pytest.raises(ValueError, match="Unknown protocol"):
            _detect_protocol("http://example.com/file.tif")

    @pytest.mark.unit
    def test_parse_cloud_path_gcs(self):
        """Test parsing GCS cloud path."""
        protocol, bucket, key = _parse_cloud_path("gs://my-bucket/path/file.tif")

        assert protocol == "gs"
        assert bucket == "my-bucket"
        assert key == "path/file.tif"

    @pytest.mark.unit
    def test_parse_cloud_path_s3(self):
        """Test parsing S3 cloud path."""
        protocol, bucket, key = _parse_cloud_path("s3://my-bucket/path/file.tif")

        assert protocol == "s3"
        assert bucket == "my-bucket"
        assert key == "path/file.tif"


class TestAEFTileInfoSource:
    """Tests for AEFTileInfo source field."""

    @pytest.mark.unit
    def test_tile_info_with_gcs_source(self, sample_gcs_tile_info):
        """Test AEFTileInfo with GCS source."""
        assert sample_gcs_tile_info.source == DataSource.GCS
        assert not sample_gcs_tile_info.is_source_coop

    @pytest.mark.unit
    def test_tile_info_with_source_coop(self, sample_source_coop_tile_info):
        """Test AEFTileInfo with Source Coop source."""
        assert sample_source_coop_tile_info.source == DataSource.SOURCE_COOP
        assert sample_source_coop_tile_info.is_source_coop

    @pytest.mark.unit
    def test_tile_info_without_source(self):
        """Test AEFTileInfo without source field."""
        tile = AEFTileInfo(
            id="test",
            path="gs://bucket/file.tif",
            year=2023,
            bbox=(-122.5, 37.5, -122.0, 38.0),
            crs_epsg=32610,
        )
        assert tile.source is None
        assert not tile.is_source_coop


class TestAEFIndexSourceCoop:
    """Tests for AEFIndex with Source Cooperative source."""

    @pytest.mark.unit
    def test_index_init_source_coop(self):
        """Test AEFIndex initialization with Source Coop source."""
        index = AEFIndex(source=DataSource.SOURCE_COOP)

        assert index.source == DataSource.SOURCE_COOP
        assert index._bucket == SOURCE_COOP_BUCKET
        assert index._cache_filename == "aef_index_source_coop.parquet"

    @pytest.mark.unit
    def test_index_init_gcs(self):
        """Test AEFIndex initialization with GCS source."""
        index = AEFIndex(source=DataSource.GCS, gcp_project="my-project")

        assert index.source == DataSource.GCS
        assert index._bucket == GCS_BUCKET
        assert index._cache_filename == "aef_index_gcs.parquet"

    @pytest.mark.unit
    def test_index_default_source_is_gcs(self):
        """Test that default source is GCS."""
        index = AEFIndex()

        assert index.source == DataSource.GCS

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_source_coop_download_no_project_required(self, tmp_path):
        """Test that Source Coop download does not require GCP project."""
        # Create a cached file to avoid actual download
        cache_file = tmp_path / "aef_index_source_coop.parquet"
        cache_file.touch()

        index = AEFIndex(source=DataSource.SOURCE_COOP, cache_dir=tmp_path)

        # Should not raise - Source Coop doesn't need GCP project
        result = await index.download()

        assert result == cache_file

    @pytest.mark.unit
    def test_convert_path_to_source_coop(self):
        """Test path conversion for Source Coop."""
        index = AEFIndex(source=DataSource.SOURCE_COOP)

        gcs_path = "gs://alphaearth_foundations/satellite_embedding/v1/annual/2023/10N/tile.tiff"
        expected = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2023/10N/tile.tiff"

        result = index._convert_path_to_source(gcs_path)

        assert result == expected

    @pytest.mark.unit
    def test_convert_path_to_gcs(self):
        """Test path conversion keeps GCS path unchanged."""
        index = AEFIndex(source=DataSource.GCS, gcp_project="my-project")

        gcs_path = "gs://alphaearth_foundations/satellite_embedding/v1/annual/2023/10N/tile.tiff"

        result = index._convert_path_to_source(gcs_path)

        assert result == gcs_path

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_populates_source(self, tmp_path, mock_source_coop_gdf):
        """Test that query populates source field on tiles."""
        # Save mock gdf
        index_path = tmp_path / "aef_index_source_coop.parquet"
        mock_source_coop_gdf.to_parquet(index_path)

        index = AEFIndex(source=DataSource.SOURCE_COOP, cache_dir=tmp_path)
        index.load()

        tiles = await index.query(limit=1)

        assert len(tiles) == 1
        assert tiles[0].source == DataSource.SOURCE_COOP
        assert tiles[0].path.startswith("s3://")
