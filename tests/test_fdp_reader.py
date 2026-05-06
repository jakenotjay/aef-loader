"""Tests for aef_loader.fdp.reader module."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from aef_loader.fdp.reader import FDPReader
from aef_loader.types import FDPTileInfo
from xarray import DataTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_tile_ds(
    *,
    x_start: float,
    n_x: int,
    y_start: float,
    n_y: int,
    res: float = 0.5,
    year: int = 2024,
    fill: float = 0.5,
) -> xr.Dataset:
    """Build an in-memory dataset shaped like an opened FDP tile.

    Northern-up (y decreases). Spatial coords are pixel-centre style — exact
    layout doesn't matter for the combine logic, only that contiguous tiles
    have non-overlapping coords on the same grid.
    """
    x = x_start + np.arange(n_x, dtype=np.float64) * res
    y = y_start - np.arange(n_y, dtype=np.float64) * res
    data = np.full((1, n_y, n_x), fill, dtype=np.float32)
    ds = xr.Dataset(
        {"probability": (("time", "y", "x"), data)},
        coords={
            "time": [np.datetime64(f"{year}-01-01")],
            "y": y,
            "x": x,
        },
    )
    ds["probability"].attrs.update({"_FillValue": np.nan, "nodata": np.nan})
    return ds


def _fake_tile_info(
    commodity: str = "coffee",
    year: int = 2024,
    lng: int = 9,
    lat: int = 5,
) -> FDPTileInfo:
    return FDPTileInfo(
        id=f"{commodity}_{year}_lng_{lng}_lat_{lat}",
        path=(
            f"gs://earth-engine-public-requester-pays/forestdatapartnership/"
            f"2025b/{commodity}/{year}/lng_{lng}_lat_{lat}.tif"
        ),
        year=year,
        bbox=(float(lng), float(lat), float(lng + 1), float(lat + 1)),
        commodity=commodity,  # type: ignore[arg-type]
        release="2025b",
    )


# ---------------------------------------------------------------------------
# Context manager + store wiring
# ---------------------------------------------------------------------------


class TestFDPReader:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_manager_clears_state(self):
        async with FDPReader() as reader:
            reader._stores["gs://x"] = MagicMock()
            reader._registry = MagicMock()
            assert reader is not None

        assert reader._stores == {}
        assert reader._registry is None

    @pytest.mark.unit
    def test_get_store_passes_project_to_make_gcs_store(self):
        reader = FDPReader(gcp_project="my-project")
        with patch("aef_loader.fdp.reader.make_gcs_store") as mock_make:
            reader._get_store("gs", "my-bucket")
            mock_make.assert_called_once_with("my-bucket", "my-project")

    @pytest.mark.unit
    def test_get_store_without_project_raises(self):
        reader = FDPReader()
        with pytest.raises(ValueError, match="gcp_project is required"):
            reader._get_store("gs", "my-bucket")

    @pytest.mark.unit
    def test_get_store_caches(self):
        reader = FDPReader(gcp_project="p")
        sentinel = MagicMock()
        reader._stores["gs://my-bucket"] = sentinel
        with patch("aef_loader.fdp.reader.make_gcs_store") as mock_make:
            store = reader._get_store("gs", "my-bucket")
            mock_make.assert_not_called()
        assert store is sentinel

    @pytest.mark.unit
    def test_get_store_rejects_non_gs_protocol(self):
        reader = FDPReader(gcp_project="p")
        with pytest.raises(ValueError, match="only supports gs"):
            reader._get_store("s3", "my-bucket")


# ---------------------------------------------------------------------------
# _combine_opened_datasets — pure xarray, no IO
# ---------------------------------------------------------------------------


class TestCombineOpenedDatasets:
    @pytest.mark.unit
    def test_outer_join_unions_contiguous_tiles(self):
        # Tile A: x=[9.0, 9.5, 10.0, 10.5]; Tile B: x=[11.0, 11.5, 12.0, 12.5].
        # Same y grid, both 2024 — coords are contiguous on a 0.5° step
        # (10.5 → 11.0), so the outer join produces a clean union with no
        # NaN-padded gap. The disjoint-coords case is exercised separately
        # by test_outer_join_introduces_nan_for_offset_tiles below.
        ds_a = _make_synthetic_tile_ds(
            x_start=9.0, n_x=4, y_start=6.0, n_y=2, fill=0.25
        )
        ds_b = _make_synthetic_tile_ds(
            x_start=11.0, n_x=4, y_start=6.0, n_y=2, fill=0.75
        )

        combined = FDPReader._combine_opened_datasets([ds_a, ds_b])

        assert list(combined.coords["x"].values) == [
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
            12.5,
        ]
        assert combined.sizes["time"] == 1
        np.testing.assert_array_equal(
            combined["probability"].isel(time=0, y=0).values,
            np.array([0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75]),
        )
        assert np.isnan(combined["probability"].attrs["_FillValue"])

    @pytest.mark.unit
    def test_outer_join_introduces_nan_for_offset_tiles(self):
        # Tile A covers x=[9.0, 9.5, 10.0, 10.5] at y=[6.0, 5.5].
        # Tile B covers x=[11.0, 11.5] at y=[5.5, 5.0] — partially overlapping
        # in y but at completely different x. Outer-join produces a 6-wide,
        # 3-tall grid with NaNs in the corners that no tile covers.
        ds_a = _make_synthetic_tile_ds(
            x_start=9.0, n_x=4, y_start=6.0, n_y=2, fill=0.25
        )
        ds_b = _make_synthetic_tile_ds(
            x_start=11.0, n_x=2, y_start=5.5, n_y=2, fill=0.75
        )

        combined = FDPReader._combine_opened_datasets([ds_a, ds_b])

        xs = list(combined.coords["x"].values)
        ys = list(combined.coords["y"].values)
        assert xs == [9.0, 9.5, 10.0, 10.5, 11.0, 11.5]
        # combine_by_coords sorts coords ascending: [5.0, 5.5, 6.0].
        assert ys == [5.0, 5.5, 6.0]

        arr = combined["probability"].isel(time=0).values
        y_to_row = {y: i for i, y in enumerate(ys)}
        # y=6.0: A covers 0..3 with 0.25; no tile covers 4..5 → NaN.
        top = arr[y_to_row[6.0]]
        assert np.allclose(top[:4], 0.25)
        assert all(np.isnan(top[4:]))
        # y=5.0: no tile covers 0..3; B covers 4..5 with 0.75.
        bot = arr[y_to_row[5.0]]
        assert all(np.isnan(bot[:4]))
        assert np.allclose(bot[4:], 0.75)

    @pytest.mark.unit
    def test_two_year_concat_produces_time_dim(self):
        ds_2023 = _make_synthetic_tile_ds(
            x_start=9.0, n_x=4, y_start=6.0, n_y=2, year=2023, fill=0.1
        )
        ds_2024 = _make_synthetic_tile_ds(
            x_start=9.0, n_x=4, y_start=6.0, n_y=2, year=2024, fill=0.9
        )

        combined = FDPReader._combine_opened_datasets([ds_2024, ds_2023])

        assert combined.sizes["time"] == 2
        # Sorted ascending: 2023 first, 2024 second.
        assert combined["probability"].isel(time=0, y=0, x=0).item() == pytest.approx(
            0.1
        )
        assert combined["probability"].isel(time=1, y=0, x=0).item() == pytest.approx(
            0.9
        )


# ---------------------------------------------------------------------------
# open() — tile grouping behaviour
# ---------------------------------------------------------------------------


class TestOpen:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_tiles_raises(self):
        async with FDPReader() as reader:
            with pytest.raises(ValueError, match="No tiles provided"):
                await reader.open([])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_commodity_returns_datatree(self):
        """Critical: single-commodity input MUST still return a DataTree."""
        tiles = [_fake_tile_info(commodity="coffee", lng=9, lat=5)]

        async def fake_open_tile(self, tile, parser, chunks):
            return _make_synthetic_tile_ds(
                x_start=float(tile.lng),
                n_x=4,
                y_start=float(tile.lat + 1),
                n_y=2,
                year=tile.year,
            )

        async with FDPReader() as reader:
            with patch.object(FDPReader, "_open_tile", new=fake_open_tile):
                tree = await reader.open(tiles)

        assert isinstance(tree, DataTree)
        assert list(tree.children) == ["coffee"]
        assert "probability" in tree["/coffee"].ds.data_vars
        assert tree.attrs["total_tiles"] == 1
        assert tree.attrs["commodities"] == ["coffee"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mixed_commodities_yields_one_child_each(self):
        tiles = [
            _fake_tile_info(commodity="coffee", lng=9, lat=5),
            _fake_tile_info(commodity="cocoa", lng=9, lat=5),
            _fake_tile_info(commodity="coffee", lng=10, lat=5),
        ]

        async def fake_open_tile(self, tile, parser, chunks):
            return _make_synthetic_tile_ds(
                x_start=float(tile.lng),
                n_x=2,
                y_start=float(tile.lat + 1),
                n_y=2,
                year=tile.year,
                fill={"coffee": 0.3, "cocoa": 0.7}[tile.commodity],
            )

        async with FDPReader() as reader:
            with patch.object(FDPReader, "_open_tile", new=fake_open_tile):
                tree = await reader.open(tiles)

        assert isinstance(tree, DataTree)
        assert set(tree.children) == {"coffee", "cocoa"}
        # coffee has two adjacent tiles → wider x extent than cocoa.
        coffee_xs = tree["/coffee"].ds.sizes["x"]
        cocoa_xs = tree["/cocoa"].ds.sizes["x"]
        assert coffee_xs > cocoa_xs
        assert tree.attrs["total_tiles"] == 3
        assert set(tree.attrs["commodities"]) == {"coffee", "cocoa"}


# ---------------------------------------------------------------------------
# Integration — hits real GCS. Skip cleanly without creds.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.requires_gcp
class TestFDPReaderIntegration:
    """Run with: GCP_PROJECT=epoch-geospatial-dev uv run pytest -m slow"""

    @pytest.fixture(scope="class")
    def gcp_project(self):
        project = os.environ.get("GCP_PROJECT")
        if project is None:
            pytest.skip("GCP_PROJECT not set — skipping FDP reader integration test")
        return project

    @pytest.mark.asyncio
    async def test_open_two_adjacent_cameroon_tiles(self, gcp_project):
        from aef_loader.fdp.index import FDPIndex

        cache_dir = Path.home() / ".cache" / "aef-loader"
        index = FDPIndex(release="2025b", gcp_project=gcp_project, cache_dir=cache_dir)
        if not (cache_dir / "fdp_index_2025b.parquet").exists():
            await index.build()
        index.load()

        # Strictly inside the (9,5)+(10,5) tile pair, just past the x=10
        # shared edge — avoids edge-touching neighbours under shapely's
        # intersects semantics.
        tiles = await index.query(
            bbox=(9.1, 5.1, 10.9, 5.9),
            years=2024,
            commodities=["coffee"],
        )
        assert len(tiles) == 2
        assert {(t.lng, t.lat) for t in tiles} == {(9, 5), (10, 5)}

        async with FDPReader(gcp_project=gcp_project) as reader:
            tree = await reader.open(tiles)

        assert isinstance(tree, DataTree)
        assert list(tree.children) == ["coffee"]

        ds = tree["/coffee"].ds
        assert ds["probability"].dtype == np.float32

        xs = ds.coords["x"].values
        ys = ds.coords["y"].values
        assert xs.min() >= 9.0 - 1e-3 and xs.max() <= 11.0 + 1e-3
        assert ys.min() >= 5.0 - 1e-3 and ys.max() <= 6.0 + 1e-3

        sample = (
            ds["probability"]
            .isel(time=0, y=slice(0, 256), x=slice(0, 256))
            .compute()
            .values
        )
        if np.isfinite(sample).any():
            assert float(np.nanmin(sample)) >= 0.0
            assert float(np.nanmax(sample)) <= 1.0
