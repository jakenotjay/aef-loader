"""Pytest fixtures for aef-loader tests."""

import numpy as np
import pytest
import xarray as xr
from aef_loader.constants import DataSource
from aef_loader.types import AEFTileInfo
from shapely.geometry import box


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "unit: Unit tests that do not require GCP credentials (fast, isolated)",
    )
    config.addinivalue_line(
        "markers",
        "integration: Integration tests that may require GCP credentials (slower, external dependencies)",
    )
    config.addinivalue_line(
        "markers",
        "requires_gcp: Tests that require authenticated GCP client access",
    )
    config.addinivalue_line(
        "markers",
        "requires_aws: Tests that require AWS S3 access (Source Cooperative)",
    )


@pytest.fixture
def sample_bbox():
    """Sample bounding box for testing."""
    return (-122.5, 37.5, -122.0, 38.0)


@pytest.fixture
def sample_polygon(sample_bbox):
    """Sample polygon for testing."""
    return box(*sample_bbox)


@pytest.fixture
def sample_tile_info():
    """Sample AEFTileInfo for testing."""
    return AEFTileInfo(
        id="test_tile_001",
        path="gs://alphaearth_foundations/test/tile.tif",
        year=2023,
        bbox=(-122.5, 37.5, -122.0, 38.0),
        crs_epsg=32610,  # UTM 10N
        utm_zone="10N",
    )


@pytest.fixture
def sample_embedding_array():
    """Sample 64-channel embedding array for testing (int8, excluding -128 nodata)."""
    # Shape: (channels, height, width)
    # Range [-127, 127] excludes -128 which is the NoData value
    return np.random.randint(-127, 128, size=(64, 256, 256), dtype=np.int8)


@pytest.fixture
def sample_xarray_dataset(sample_embedding_array):
    """Sample xarray Dataset for testing."""
    data = sample_embedding_array.astype(np.float32)

    ds = xr.Dataset(
        {
            "embeddings": (["band", "y", "x"], data),
        },
        coords={
            "band": np.arange(64),
            "y": np.linspace(4200000, 4197440, 256),  # 10m resolution, decreasing
            "x": np.linspace(500000, 502560, 256),  # 10m resolution, increasing
        },
        attrs={
            "crs": "EPSG:32610",
            "source": "test",
        },
    )
    return ds


@pytest.fixture
def temp_zarr_path(tmp_path):
    """Temporary path for zarr output."""
    return tmp_path / "test_output.zarr"


@pytest.fixture
def mock_gdf():
    """Mock GeoDataFrame for index testing."""
    import geopandas as gpd
    from shapely.geometry import box

    data = {
        "fid": [1, 2, 3],
        "path": [
            "gs://alphaearth_foundations/tile1.tif",
            "gs://alphaearth_foundations/tile2.tif",
            "gs://alphaearth_foundations/tile3.tif",
        ],
        "year": [2022, 2023, 2023],
        "wgs84_west": [-122.5, -122.3, -122.1],
        "wgs84_south": [37.5, 37.5, 37.5],
        "wgs84_east": [-122.3, -122.1, -121.9],
        "wgs84_north": [37.7, 37.7, 37.7],
        "utm_west": [500000.0, 518192.0, 536384.0],
        "utm_south": [4152000.0, 4152000.0, 4152000.0],
        "utm_east": [518192.0, 536384.0, 554576.0],
        "utm_north": [4174192.0, 4174192.0, 4174192.0],
        "crs": ["EPSG:32610", "EPSG:32610", "EPSG:32610"],
        "utm_zone": ["10N", "10N", "10N"],
        "geometry": [
            box(-122.5, 37.5, -122.3, 37.7),
            box(-122.3, 37.5, -122.1, 37.7),
            box(-122.1, 37.5, -121.9, 37.7),
        ],
    }

    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def sample_source_coop_tile_info():
    """Sample AEFTileInfo for Source Cooperative testing."""
    return AEFTileInfo(
        id="test_tile_source_coop_001",
        path="s3://tge-labs-aef/v1/annual/2023/10N/test_tile.tiff",
        year=2023,
        bbox=(-122.5, 37.5, -122.0, 38.0),
        crs_epsg=32610,  # UTM 10N
        utm_zone="10N",
        utm_bounds=(500000.0, 4152000.0, 581920.0, 4233920.0),
        source=DataSource.SOURCE_COOP,
    )


@pytest.fixture
def sample_gcs_tile_info():
    """Sample AEFTileInfo for GCS testing."""
    return AEFTileInfo(
        id="test_tile_gcs_001",
        path="gs://alphaearth_foundations/satellite_embedding/v1/annual/2023/10N/test_tile.tiff",
        year=2023,
        bbox=(-122.5, 37.5, -122.0, 38.0),
        crs_epsg=32610,  # UTM 10N
        utm_zone="10N",
        utm_bounds=(500000.0, 4152000.0, 581920.0, 4233920.0),
        source=DataSource.GCS,
    )


@pytest.fixture
def mock_source_coop_gdf():
    """Mock GeoDataFrame for Source Cooperative index testing."""
    import geopandas as gpd
    from shapely.geometry import box

    data = {
        "fid": [1, 2, 3],
        "path": [
            "gs://alphaearth_foundations/satellite_embedding/v1/annual/2022/10N/tile1.tiff",
            "gs://alphaearth_foundations/satellite_embedding/v1/annual/2023/10N/tile2.tiff",
            "gs://alphaearth_foundations/satellite_embedding/v1/annual/2023/10N/tile3.tiff",
        ],
        "year": [2022, 2023, 2023],
        "wgs84_west": [-122.5, -122.3, -122.1],
        "wgs84_south": [37.5, 37.5, 37.5],
        "wgs84_east": [-122.3, -122.1, -121.9],
        "wgs84_north": [37.7, 37.7, 37.7],
        "utm_west": [500000.0, 518192.0, 536384.0],
        "utm_south": [4152000.0, 4152000.0, 4152000.0],
        "utm_east": [518192.0, 536384.0, 554576.0],
        "utm_north": [4174192.0, 4174192.0, 4174192.0],
        "crs": ["EPSG:32610", "EPSG:32610", "EPSG:32610"],
        "utm_zone": ["10N", "10N", "10N"],
        "geometry": [
            box(-122.5, 37.5, -122.3, 37.7),
            box(-122.3, 37.5, -122.1, 37.7),
            box(-122.1, 37.5, -121.9, 37.7),
        ],
    }

    return gpd.GeoDataFrame(data, crs="EPSG:4326")
