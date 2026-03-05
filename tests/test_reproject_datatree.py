"""Integration test for reproject_datatree across two UTM zones."""

from __future__ import annotations

import numpy as np
import pytest
from aef_loader.constants import DataSource
from aef_loader.index import AEFIndex
from aef_loader.reader import VirtualTiffReader
from aef_loader.utils import reproject_datatree
from odc.geo.geobox import GeoBox


# Bbox that straddles the UTM 10N / 11N boundary at -120° longitude.
# Western edge in zone 10N (EPSG:32610), eastern edge in zone 11N (EPSG:32611).
CROSS_ZONE_BBOX = (-120.1, 38.9, -119.9, 39.1)
YEAR = 2021


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("reproject_cache")


@pytest.fixture(scope="module")
def source_coop_index(cache_dir):
    return AEFIndex(source=DataSource.SOURCE_COOP, cache_dir=cache_dir)


@pytest.mark.integration
class TestReprojectDatatreeCrossZone:
    """Integration test: load tiles spanning two UTM zones, reproject, verify chunks."""

    @pytest.mark.asyncio
    async def test_cross_zone_reproject_preserves_band_chunks(self, source_coop_index):
        """Load real tiles crossing UTM 10N/11N, reproject, and verify band chunks stay intact."""
        await source_coop_index.download()
        source_coop_index.load()
        tiles = await source_coop_index.query(bbox=CROSS_ZONE_BBOX, years=YEAR)
        assert len(tiles) > 0, "No tiles found for cross-zone bbox"

        # Confirm we actually have tiles in two different zones
        zones = {t.utm_zone for t in tiles}
        assert len(zones) >= 2, f"Expected tiles in ≥2 zones, got {zones}"

        async with VirtualTiffReader() as reader:
            tree = await reader.open_tiles_by_zone(tiles)

        assert len(tree.children) >= 2

        # Reproject to a coarse WGS84 geobox (keeps data small)
        target = GeoBox.from_bbox(
            bbox=CROSS_ZONE_BBOX,
            crs="EPSG:4326",
            resolution=0.005,  # ~500m — fast to compute
        )

        combined = reproject_datatree(tree, target)

        # Band dimension must be a single chunk (the bug was fragmentation here)
        chunks = combined["embeddings"].chunks
        assert chunks is not None, "Expected dask-backed array with chunks"
        band_chunks = chunks[0]
        assert band_chunks == (64,), (
            f"Band dim fragmented into {band_chunks}, expected (64,)"
        )

        # Sanity-check the computed result
        result = combined.isel(time=0).compute()
        emb = result["embeddings"]

        assert emb.dims == ("band", "y", "x")
        assert emb.sizes["band"] == 64
        assert emb.dtype == np.int8

        # At least some pixels should be non-nodata (-128)
        valid_pixels = (emb.values != -128).sum()
        assert valid_pixels > 0, "All pixels are nodata — reprojection may have failed"
