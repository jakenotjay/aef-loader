"""Integration tests for tight-geobox reprojection optimisation.

Uses real AEF tiles from Source Cooperative (public S3, no auth needed).

Run with:
    cd packages/aef-loader
    pytest tests/test_reprojection.py -m integration -v
"""

from __future__ import annotations

import numpy as np
import pytest
from aef_loader.constants import DataSource
from aef_loader.index import AEFIndex
from aef_loader.reader import VirtualTiffReader
from aef_loader.utils import reproject_datatree
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_reproject

# Small bbox in the San Francisco area (fits within a single UTM zone 10N)
BBOX = (-122.5, 37.5, -122.0, 38.0)
YEAR = 2023


# ---------------------------------------------------------------------------
# Module-scoped fixtures — shared across all tests in the file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    """Shared temp dir so AEFIndex.download() caches across tests."""
    return tmp_path_factory.mktemp("aef_reprojection_cache")


@pytest.fixture(scope="module")
def source_coop_index(cache_dir):
    """Ready-to-use AEFIndex backed by Source Cooperative."""
    index = AEFIndex(source=DataSource.SOURCE_COOP, cache_dir=cache_dir)
    # download + load synchronously so every test can query immediately
    import asyncio

    asyncio.get_event_loop().run_until_complete(index.download())
    index.load()
    return index


@pytest.fixture(scope="module")
def tiles(source_coop_index):
    """A handful of tiles covering BBOX."""
    import asyncio

    return asyncio.get_event_loop().run_until_complete(
        source_coop_index.query(bbox=BBOX, years=YEAR, limit=4)
    )


@pytest.fixture(scope="module")
def datatree(tiles):
    """DataTree with zone children, built from the queried tiles."""
    import asyncio

    async def _open():
        async with VirtualTiffReader() as reader:
            return await reader.open_tiles_by_zone(tiles)

    return asyncio.get_event_loop().run_until_complete(_open())


@pytest.fixture(scope="module")
def target_geobox():
    """A WGS-84 target GeoBox covering BBOX at ~100 m resolution."""
    return GeoBox.from_bbox(bbox=BBOX, crs="EPSG:4326", resolution=0.001)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTightGeoboxReprojection:
    """Tests for the tight-geobox reprojection optimisation."""

    def test_reproject_single_zone(self, datatree, target_geobox):
        """Reprojecting a single zone to a WGS-84 target produces valid data."""
        combined = reproject_datatree(datatree, target_geobox)
        chunk = combined["embeddings"].sel(band="A00").compute()

        # Should have non-NaN data somewhere
        assert not np.all(np.isnan(chunk.values)), "All values are NaN"

        # CRS should be set on the result
        assert combined.odc.crs is not None
        assert combined.odc.crs.epsg == 4326

    def test_reproject_values_preserved(self, datatree, target_geobox):
        """Nearest-neighbour reprojection keeps int8 values in the expected range."""
        combined = reproject_datatree(datatree, target_geobox, resampling="nearest")
        chunk = combined["embeddings"].sel(band="A00").compute()

        valid = chunk.values[~np.isnan(chunk.values)]
        assert len(valid) > 0, "No valid pixels after reprojection"

        # int8 AEF values are in [-128, 127]; after reprojection NaN-filled
        # pixels become float NaN, but valid pixels should be in range.
        assert valid.min() >= -128
        assert valid.max() <= 127

        # Should not be all zeros
        assert not np.all(valid == 0), "All valid pixels are zero"

    def test_canonical_coordinates_no_duplicates(self, datatree, target_geobox):
        """Combined result has strictly monotonic row and column coordinates."""
        combined = reproject_datatree(datatree, target_geobox)

        # Dimension names depend on CRS (y/x for projected, latitude/longitude for geographic)
        row_dim, col_dim = target_geobox.dimensions
        row_vals = combined.coords[row_dim].values
        col_vals = combined.coords[col_dim].values

        # Row coordinate should be strictly monotonic (decreasing for top-down grids)
        row_diffs = np.diff(row_vals)
        assert np.all(row_diffs < 0) or np.all(row_diffs > 0), (
            f"{row_dim} coordinates are not strictly monotonic"
        )

        # Column coordinate should be strictly monotonic (increasing)
        col_diffs = np.diff(col_vals)
        assert np.all(col_diffs > 0) or np.all(col_diffs < 0), (
            f"{col_dim} coordinates are not strictly monotonic"
        )

        # No duplicate values
        assert len(np.unique(row_vals)) == len(row_vals), (
            f"Duplicate {row_dim} coordinates found"
        )
        assert len(np.unique(col_vals)) == len(col_vals), (
            f"Duplicate {col_dim} coordinates found"
        )

    def test_single_var_fewer_dask_tasks_than_split(self, datatree, target_geobox):
        """Single 'embeddings' variable produces far fewer dask tasks than 64 separate vars."""
        from aef_loader.utils import split_bands

        # New approach: reproject single embeddings variable
        single_var_ds = reproject_datatree(datatree, target_geobox)
        single_var_tasks = sum(
            len(single_var_ds[v].data.__dask_graph__()) for v in single_var_ds.data_vars
        )

        # Old approach: split into 64 variables, then reproject
        split_datasets = []
        for zone_name in datatree.children:
            zone_ds = datatree[zone_name].ds
            if zone_ds is None or len(zone_ds.data_vars) == 0:
                continue
            split_ds = split_bands(zone_ds)
            split_datasets.append(
                xr_reproject(split_ds, target_geobox, resampling="nearest")
            )

        if len(split_datasets) == 1:
            split_combined = split_datasets[0]
        else:
            split_combined = split_datasets[0]
            for ds in split_datasets[1:]:
                split_combined = split_combined.combine_first(ds)

        split_tasks = sum(
            len(split_combined[v].data.__dask_graph__())
            for v in split_combined.data_vars
        )

        assert single_var_tasks < split_tasks, (
            f"Single-variable approach ({single_var_tasks} tasks) should have fewer "
            f"tasks than 64-variable approach ({split_tasks} tasks)"
        )
