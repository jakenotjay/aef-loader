"""Pytest fixtures for aef-loader tests."""

import pytest


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
    config.addinivalue_line(
        "markers",
        "slow: Slow tests that hit real cloud data. Run manually with: uv run pytest -m slow",
    )


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
def mock_fdp_gdf():
    """Mock GeoDataFrame for FDP index testing.

    Six tiles spanning two commodities × two years across a small bbox in
    Cameroon (a coffee/cocoa region).
    """
    import geopandas as gpd
    from shapely.geometry import box

    rows = []
    for commodity in ("coffee", "cocoa"):
        for year in (2020, 2024):
            for lng, lat in [(9, 5), (10, 5), (11, 6)]:
                rows.append(
                    {
                        "id": f"{commodity}_{year}_lng_{lng}_lat_{lat}",
                        "release": "2025b",
                        "commodity": commodity,
                        "year": year,
                        "lng": lng,
                        "lat": lat,
                        "path": (
                            f"gs://earth-engine-public-requester-pays/"
                            f"forestdatapartnership/2025b/{commodity}/{year}/"
                            f"lng_{lng}_lat_{lat}.tif"
                        ),
                        "geometry": box(lng, lat, lng + 1, lat + 1),
                    }
                )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


@pytest.fixture
def mock_source_coop_gdf():
    """Mock GeoDataFrame for Source Cooperative index testing."""
    import geopandas as gpd
    from shapely.geometry import box

    data = {
        "fid": [1, 2, 3],
        "path": [
            "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2022/10N/tile1.tiff",
            "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2023/10N/tile2.tiff",
            "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2023/10N/tile3.tiff",
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
