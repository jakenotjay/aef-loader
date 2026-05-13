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
    async def source_coop_index(self, cache_dir):
        index = AEFIndex(source=DataSource.SOURCE_COOP, cache_dir=cache_dir)
        await index.download()
        index.load()
        return index

    @pytest.mark.asyncio
    async def test_download_index(self, source_coop_index):
        """Index was downloaded and loaded by fixture — file exists and is non-empty."""
        path = source_coop_index._index_path
        assert path is not None
        assert path.exists()
        assert path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_query_index(self, source_coop_index):
        """Query bbox+year — tiles have s3:// paths and SOURCE_COOP source."""
        tiles = await source_coop_index.query(bbox=BBOX, years=YEAR)

        assert len(tiles) > 0
        for tile in tiles:
            assert tile.path.startswith("s3://")
            assert tile.source == DataSource.SOURCE_COOP
            assert tile.year == YEAR

    @pytest.mark.asyncio
    async def test_virtual_load_tile(self, source_coop_index):
        """Open 1 tile via open_tiles_by_zone — DataTree with zone children, single embeddings var."""
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
    async def test_selective_band_loading(self, source_coop_index):
        """Load only bands [0, 3, 63] — dataset has exactly those bands, correctly named."""
        tiles = await source_coop_index.query(bbox=BBOX, years=YEAR, limit=1)
        assert len(tiles) == 1

        selected = [0, 3, 63]
        async with VirtualTiffReader() as reader:
            tree = await reader.open_tiles_by_zone(tiles, bands=selected)

        zone_name = list(tree.children.keys())[0]
        ds = tree[zone_name].ds

        assert "embeddings" in ds.data_vars
        assert ds.sizes["band"] == 3
        assert list(ds.coords["band"].values) == ["A00", "A03", "A63"]

        # Compute a small chunk to verify data is actually readable
        chunk = (
            ds["embeddings"]
            .sel(band="A03")
            .isel(time=0, y=slice(0, 256), x=slice(0, 256))
            .compute()
        )
        assert chunk.shape == (256, 256)
        assert chunk.dtype == np.int8

    @pytest.mark.asyncio
    async def test_selective_band_loading_by_name(self, source_coop_index):
        """Load bands by name — same result as by index."""
        tiles = await source_coop_index.query(bbox=BBOX, years=YEAR, limit=1)
        assert len(tiles) == 1

        async with VirtualTiffReader() as reader:
            tree = await reader.open_tiles_by_zone(tiles, bands=["A00", "A03", "A63"])

        zone_name = list(tree.children.keys())[0]
        ds = tree[zone_name].ds

        assert ds.sizes["band"] == 3
        assert list(ds.coords["band"].values) == ["A00", "A03", "A63"]

    @pytest.mark.asyncio
    async def test_selective_bands_match_full_load(self, source_coop_index):
        """Selected bands produce identical data to loading all 64 then slicing."""
        tiles = await source_coop_index.query(bbox=BBOX, years=YEAR, limit=1)
        assert len(tiles) == 1

        selected = [0, 3, 63]
        band_names = ["A00", "A03", "A63"]
        roi = dict(time=0, y=slice(0, 256), x=slice(0, 256))

        async with VirtualTiffReader() as reader:
            tree_full = await reader.open_tiles_by_zone(tiles)
            tree_subset = await reader.open_tiles_by_zone(tiles, bands=selected)

        zone = list(tree_full.children.keys())[0]
        full_ds = tree_full[zone].ds
        subset_ds = tree_subset[zone].ds

        for name in band_names:
            full_chunk = full_ds["embeddings"].sel(band=name).isel(**roi).compute()
            subset_chunk = subset_ds["embeddings"].sel(band=name).isel(**roi).compute()
            np.testing.assert_array_equal(full_chunk.values, subset_chunk.values)


# ---------------------------------------------------------------------------
# GCS (requester-pays)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_gcp
class TestGCSIntegration:
    """Integration tests against the GCS requester-pays backend."""

    @pytest.fixture(scope="class")
    async def gcs_index(self, cache_dir, gcp_project):
        index = AEFIndex(
            source=DataSource.GCS,
            gcp_project=gcp_project,
            cache_dir=cache_dir,
        )
        await index.download()
        index.load()
        return index

    @pytest.mark.asyncio
    async def test_download_index(self, gcs_index):
        """Index was downloaded and loaded by fixture — file exists and is non-empty."""
        path = gcs_index._index_path
        assert path is not None
        assert path.exists()
        assert path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_query_index(self, gcs_index):
        """Query bbox+year — tiles have gs:// paths and GCS source."""
        tiles = await gcs_index.query(bbox=BBOX, years=YEAR)

        assert len(tiles) > 0
        for tile in tiles:
            assert tile.path.startswith("gs://")
            assert tile.source == DataSource.GCS
            assert tile.year == YEAR

    @pytest.mark.asyncio
    async def test_virtual_load_tile(self, gcs_index, gcp_project):
        """Open 1 tile via open_tiles_by_zone — DataTree with zone children, single embeddings var."""
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
