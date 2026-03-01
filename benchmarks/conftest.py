"""Shared fixtures for aef-loader benchmarks.

Session-scoped so the index is downloaded once and tiles are queried once
across all benchmark files.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from odc.geo.geobox import GeoBox

from aef_loader.constants import SOURCE_COOP_BUCKET, DataSource
from aef_loader.index import AEFIndex

# Mulanje, Malawi
BENCH_BBOX = (35.36911, -16.19885, 35.864868, -15.800182)
BENCH_YEAR = 2023

# Tile limits for each benchmark type
SINGLE_TILE_LIMIT = 1
COMPOSITE_TILE_LIMIT = 4


def pytest_configure(config):
    """Register the benchmark marker."""
    config.addinivalue_line(
        "markers",
        "benchmark: Benchmarks comparing aef-loader against other loading tools",
    )


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory):
    """Shared temp dir so AEFIndex.download() caches for the whole session."""
    return tmp_path_factory.mktemp("aef_benchmark_cache")


@pytest.fixture(scope="session")
def source_coop_index(cache_dir) -> AEFIndex:
    """Ready-to-use AEFIndex backed by Source Cooperative (public S3)."""
    index = AEFIndex(source=DataSource.SOURCE_COOP, cache_dir=cache_dir)
    asyncio.run(index.download())
    index.load()
    return index


@pytest.fixture(scope="session")
def single_tile(source_coop_index):
    """A single tile for single-tile benchmarks."""
    tiles = asyncio.run(
        source_coop_index.query(
            bbox=BENCH_BBOX, years=BENCH_YEAR, limit=SINGLE_TILE_LIMIT
        )
    )
    assert len(tiles) > 0, "No tiles found for benchmark bbox/year"
    assert len(tiles) == 1, "More than one tile found"
    return tiles


@pytest.fixture(scope="session")
def composite_tiles(source_coop_index):
    """Multiple tiles for composite benchmarks."""
    tiles = asyncio.run(
        source_coop_index.query(
            bbox=BENCH_BBOX, years=BENCH_YEAR, limit=COMPOSITE_TILE_LIMIT
        )
    )
    assert len(tiles) > 0, "No tiles found for benchmark bbox/year"
    assert len(tiles) > 1, "Only one tile found"
    return tiles


@pytest.fixture(scope="session")
def target_geobox():
    """A WGS-84 target GeoBox covering the benchmark bbox at ~100 m resolution."""
    return GeoBox.from_bbox(bbox=BENCH_BBOX, crs="EPSG:4326", resolution=0.001)


@pytest.fixture(scope="session")
def single_tile_url(single_tile):
    """HTTPS URL for the single tile, for use with rioxarray / GDAL.

    Converts the S3 path ``s3://us-west-2.opendata.source.coop/tge-labs/aef/...``
    to ``https://data.source.coop/tge-labs/aef/...`` which GDAL can read
    without AWS credentials.
    """
    tile = single_tile[0]
    path = tile.path
    if path.startswith(f"s3://{SOURCE_COOP_BUCKET}/"):
        key = path[len(f"s3://{SOURCE_COOP_BUCKET}/") :]
        return f"https://data.source.coop/{key}"
    return path


@pytest.fixture(scope="session")
def composite_tile_urls(composite_tiles):
    """HTTPS URLs for composite tiles, for use with rioxarray / GDAL."""
    urls = []
    for tile in composite_tiles:
        path = tile.path
        if path.startswith(f"s3://{SOURCE_COOP_BUCKET}/"):
            key = path[len(f"s3://{SOURCE_COOP_BUCKET}/") :]
            urls.append(f"https://data.source.coop/{key}")
        else:
            urls.append(path)
    return urls


@pytest.fixture(scope="session")
def gcp_project():
    """GCP project from env var, required for GCS/XEE benchmarks."""
    return os.environ.get("GCP_PROJECT")
