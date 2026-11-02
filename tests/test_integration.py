"""Integration tests for aef_loader — hits real cloud backends.

Run Source Coop only (public, no credentials needed):
    uv run pytest packages/aef-loader/tests/test_integration.py -m integration -v -k "SourceCoop"

Run GCS (requires credentials + project):
    GCP_PROJECT=my-project uv run pytest packages/aef-loader/tests/test_integration.py -m integration -v -k "GCS"
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from aef_loader.constants import DataSource
from aef_loader.index import AEFIndex
from aef_loader.reader import VirtualTiffReader

# Shared query parameters
BBOX = (-122.5, 37.5, -122.0, 38.0)
YEAR = 2023


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    """Shared temp dir so AEFIndex.download() caches across tests in the module."""
    return tmp_path_factory.mktemp("aef_integration_cache")


@pytest.fixture(scope="module")
def gcp_project():
    """Read GCP_PROJECT env var; skip GCS tests if unset."""
    project = os.environ.get("GCP_PROJECT")
    if project is None:
        pytest.skip("GCP_PROJECT env var not set — skipping GCS integration tests")
    return project


# ---------------------------------------------------------------------------
# Source Cooperative (S3, public)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSourceCoopIntegration:
    """Integration tests against the Source Cooperative S3 backend (public)."""

    @pytest.fixture(scope="class")
    def source_coop_index(self, cache_dir):
        return AEFIndex(source=DataSource.SOURCE_COOP, cache_dir=cache_dir)

    @pytest.mark.asyncio
    async def test_download_index(self, source_coop_index):
        """Download parquet index from S3 — file exists and is non-empty."""
        path = await source_coop_index.download()

        assert path.exists()
        assert path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_query_index(self, source_coop_index):
        """Load index, query bbox+year — tiles have s3:// paths and SOURCE_COOP source."""
        source_coop_index.load()
        tiles = await source_coop_index.query(bbox=BBOX, years=YEAR)

        assert len(tiles) > 0
        for tile in tiles:
            assert tile.path.startswith("s3://")
            assert tile.source == DataSource.SOURCE_COOP
            assert tile.year == YEAR

    @pytest.mark.asyncio
    async def test_virtual_load_tile(self, source_coop_index):
        """Open 1 tile via open_tiles_by_zone — DataTree with zone children, single embeddings var."""
        source_coop_index.load()
        tiles = await source_coop_index.query(bbox=BBOX, years=YEAR, limit=1)
        assert len(tiles) == 1

        async with VirtualTiffReader() as reader:
            tree = await reader.open_tiles_by_zone(tiles)

        # DataTree has at least one zone child
        assert len(tree.children) > 0

        for _zone_name, zone_node in tree.children.items():
            ds = zone_node.ds
            # Single embeddings variable with band dimension
            assert "embeddings" in ds.data_vars
            assert len(ds.data_vars) == 1
            assert "band" in ds.dims
            assert ds.sizes["band"] == 64
            assert list(ds.coords["band"].values) == [f"A{i:02d}" for i in range(64)]
            # Must have time, y, x dimensions
            assert "time" in ds.dims
            assert "y" in ds.dims
            assert "x" in ds.dims

    @pytest.mark.asyncio
    async def test_load_single_chunk(self, source_coop_index):
        """Compute a 256x256 chunk of embeddings band 0 — shape and int8 dtype."""
        source_coop_index.load()
        tiles = await source_coop_index.query(bbox=BBOX, years=YEAR, limit=1)
        assert len(tiles) == 1

        async with VirtualTiffReader() as reader:
            tree = await reader.open_tiles_by_zone(tiles)

        zone_name = list(tree.children.keys())[0]
        ds = tree[zone_name].ds

        chunk = (
            ds["embeddings"]
            .sel(band="A00")
            .isel(time=0, y=slice(0, 256), x=slice(0, 256))
            .compute()
        )

        assert chunk.shape == (256, 256)
        assert chunk.dtype == np.int8

    @pytest.mark.asyncio
    async def test_overview_ifd_georeferencing(self, source_coop_index):
        """ifd=3 overview opens, spans the same extent as ifd=0, and has real data."""
        source_coop_index.load()
        tiles = await source_coop_index.query(
            bbox=(101.0, 0.4, 101.2, 0.6), years=2024, limit=1
        )
        assert len(tiles) == 1

        async with VirtualTiffReader() as reader:
            full = await reader.open_tiles_by_zone(tiles, ifd=0)
            ovr = await reader.open_tiles_by_zone(tiles, ifd=3)

        zone = list(full.children.keys())[0]
        ds0, ds3 = full[zone].ds, ovr[zone].ds
        assert ds3.sizes["x"] < ds0.sizes["x"]

        def edges(ds):
            dx = abs(float(ds.x[1] - ds.x[0]))
            dy = abs(float(ds.y[1] - ds.y[0]))
            return (
                (
                    float(ds.x.min()) - dx / 2,
                    float(ds.x.max()) + dx / 2,
                    float(ds.y.min()) - dy / 2,
                    float(ds.y.max()) + dy / 2,
                ),
                dx,
                dy,
            )

        b0, _, _ = edges(ds0)
        b3, dx3, dy3 = edges(ds3)
        tol = (dx3, dx3, dy3, dy3)
        for a, b, t in zip(b0, b3, tol, strict=True):
            assert abs(a - b) <= t

        chunk = (
            ds3["embeddings"]
            .isel(time=0, y=slice(0, 64), x=slice(0, 64), band=slice(0, 4))
            .compute()
        )
        assert (chunk.values != -128).any()


# ---------------------------------------------------------------------------
# GCS (requester-pays)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_gcp
class TestGCSIntegration:
    """Integration tests against the GCS requester-pays backend."""

    @pytest.fixture(scope="class")
    def gcs_index(self, cache_dir, gcp_project):
        return AEFIndex(
            source=DataSource.GCS,
            gcp_project=gcp_project,
            cache_dir=cache_dir,
        )

    @pytest.mark.asyncio
    async def test_download_index(self, gcs_index):
        """Download parquet index from GCS — file exists and is non-empty."""
        path = await gcs_index.download()

        assert path.exists()
        assert path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_query_index(self, gcs_index):
        """Load index, query bbox+year — tiles have gs:// paths and GCS source."""
        gcs_index.load()
        tiles = await gcs_index.query(bbox=BBOX, years=YEAR)

        assert len(tiles) > 0
        for tile in tiles:
            assert tile.path.startswith("gs://")
            assert tile.source == DataSource.GCS
            assert tile.year == YEAR

    @pytest.mark.asyncio
    async def test_virtual_load_tile(self, gcs_index, gcp_project):
        """Open 1 tile via open_tiles_by_zone — DataTree with zone children, single embeddings var."""
        gcs_index.load()
        tiles = await gcs_index.query(bbox=BBOX, years=YEAR, limit=1)
        assert len(tiles) == 1

        async with VirtualTiffReader(gcp_project=gcp_project) as reader:
            tree = await reader.open_tiles_by_zone(tiles)

        assert len(tree.children) > 0

        for _zone_name, zone_node in tree.children.items():
            ds = zone_node.ds
            assert "embeddings" in ds.data_vars
            assert len(ds.data_vars) == 1
            assert "band" in ds.dims
            assert ds.sizes["band"] == 64
            assert list(ds.coords["band"].values) == [f"A{i:02d}" for i in range(64)]
            assert "time" in ds.dims
            assert "y" in ds.dims
            assert "x" in ds.dims

    @pytest.mark.asyncio
    async def test_load_single_chunk(self, gcs_index, gcp_project):
        """Compute a 256x256 chunk of embeddings band 0 — shape and int8 dtype."""
        gcs_index.load()
        tiles = await gcs_index.query(bbox=BBOX, years=YEAR, limit=1)
        assert len(tiles) == 1

        async with VirtualTiffReader(gcp_project=gcp_project) as reader:
            tree = await reader.open_tiles_by_zone(tiles)

        zone_name = list(tree.children.keys())[0]
        ds = tree[zone_name].ds

        chunk = (
            ds["embeddings"]
            .sel(band="A00")
            .isel(time=0, y=slice(0, 256), x=slice(0, 256))
            .compute()
        )

        assert chunk.shape == (256, 256)
        assert chunk.dtype == np.int8
