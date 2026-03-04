"""Tests for reproject_datatree function, specifically the band chunking fix."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from unittest.mock import patch, MagicMock
from xarray import DataTree

from aef_loader.utils import reproject_datatree


def _make_dask_dataset(shape, band_names, fill_value=np.nan, chunks=None):
    """Create a dask-backed xr.Dataset simulating a reprojected zone.

    Args:
        shape: (n_bands, ny, nx) tuple
        band_names: list of band coordinate labels
        fill_value: value to fill the array with
        chunks: chunk specification, defaults to single band chunk
    """
    n_bands, ny, nx = shape
    if chunks is None:
        chunks = (n_bands, ny, nx)

    if np.isnan(fill_value):
        data = da.full(shape, np.nan, dtype=np.float32, chunks=chunks)
    else:
        data = da.full(shape, fill_value, dtype=np.float32, chunks=chunks)

    ds = xr.Dataset(
        {"embeddings": (["band", "y", "x"], data)},
        coords={
            "band": band_names,
            "y": np.arange(ny, dtype=np.float64),
            "x": np.arange(nx, dtype=np.float64),
        },
    )
    return ds


def _make_datatree_and_geobox(zone_datasets, zone_names=None):
    """Create a DataTree from zone datasets and a mock GeoBox.

    Returns (tree, mock_geobox).
    """
    if zone_names is None:
        zone_names = [f"zone_{i}" for i in range(len(zone_datasets))]

    children = {}
    for name, ds in zip(zone_names, zone_datasets):
        ds.attrs["utm_zone"] = name
        children[name] = DataTree(ds)

    tree = DataTree(children=children)
    geobox = MagicMock()
    geobox.crs = "EPSG:4326"
    return tree, geobox


class TestReprojectDatatreeBandChunking:
    """Tests verifying that reproject_datatree preserves band chunk structure."""

    @pytest.mark.unit
    @patch("aef_loader.utils.xr_reproject")
    def test_single_zone_passthrough(self, mock_reproject):
        """Single zone should be returned directly without merging."""
        band_names = [f"A{i:02d}" for i in range(4)]
        ds = _make_dask_dataset((4, 10, 10), band_names, fill_value=1.0)
        mock_reproject.return_value = ds.copy()

        tree, geobox = _make_datatree_and_geobox([ds])
        result = reproject_datatree(tree, geobox)

        assert "embeddings" in result.data_vars
        mock_reproject.assert_called_once()

    @pytest.mark.unit
    @patch("aef_loader.utils.xr_reproject")
    def test_two_zones_preserves_band_chunks(self, mock_reproject):
        """Merging two zones must preserve band as a single chunk."""
        n_bands, ny, nx = 64, 20, 20
        band_names = [f"A{i:02d}" for i in range(n_bands)]

        # Zone 1: valid data on left half, NaN on right
        data1 = da.from_delayed(
            _delayed_zone_data(n_bands, ny, nx, left=True),
            shape=(n_bands, ny, nx),
            dtype=np.float32,
        ).rechunk((n_bands, ny, nx))

        # Zone 2: NaN on left half, valid data on right
        data2 = da.from_delayed(
            _delayed_zone_data(n_bands, ny, nx, left=False),
            shape=(n_bands, ny, nx),
            dtype=np.float32,
        ).rechunk((n_bands, ny, nx))

        ds1 = xr.Dataset(
            {"embeddings": (["band", "y", "x"], data1)},
            coords={
                "band": band_names,
                "y": np.arange(ny, dtype=np.float64),
                "x": np.arange(nx, dtype=np.float64),
            },
        )
        ds2 = xr.Dataset(
            {"embeddings": (["band", "y", "x"], data2)},
            coords={
                "band": band_names,
                "y": np.arange(ny, dtype=np.float64),
                "x": np.arange(nx, dtype=np.float64),
            },
        )

        mock_reproject.side_effect = [ds1, ds2]
        tree, geobox = _make_datatree_and_geobox(
            [_make_dask_dataset((n_bands, ny, nx), band_names)] * 2
        )

        result = reproject_datatree(tree, geobox)

        # Band dimension must remain a single chunk
        band_chunks = result["embeddings"].chunks[0]
        assert band_chunks == (n_bands,), (
            f"Band dim fragmented into {band_chunks}, expected ({n_bands},)"
        )

    @pytest.mark.unit
    @patch("aef_loader.utils.xr_reproject")
    def test_fillna_semantics(self, mock_reproject):
        """Where zone 1 is NaN, zone 2's values should be used (fillna semantics)."""
        band_names = ["A00", "A01"]

        # Zone 1: row 0 valid (1.0), row 1 NaN
        arr1 = np.array(
            [[[1.0, 1.0], [np.nan, np.nan]], [[1.0, 1.0], [np.nan, np.nan]]],
            dtype=np.float32,
        )
        # Zone 2: row 0 NaN, row 1 valid (2.0)
        arr2 = np.array(
            [[[np.nan, np.nan], [2.0, 2.0]], [[np.nan, np.nan], [2.0, 2.0]]],
            dtype=np.float32,
        )

        ds1 = xr.Dataset(
            {"embeddings": (["band", "y", "x"], da.from_array(arr1, chunks=(2, 2, 2)))},
            coords={"band": band_names, "y": [0.0, 1.0], "x": [0.0, 1.0]},
        )
        ds2 = xr.Dataset(
            {"embeddings": (["band", "y", "x"], da.from_array(arr2, chunks=(2, 2, 2)))},
            coords={"band": band_names, "y": [0.0, 1.0], "x": [0.0, 1.0]},
        )

        mock_reproject.side_effect = [ds1, ds2]
        tree, geobox = _make_datatree_and_geobox(
            [_make_dask_dataset((2, 2, 2), band_names)] * 2
        )

        result = reproject_datatree(tree, geobox).compute()

        # Row 0 should come from zone 1 (1.0)
        np.testing.assert_array_equal(result["embeddings"].values[:, 0, :], 1.0)
        # Row 1 should come from zone 2 (2.0)
        np.testing.assert_array_equal(result["embeddings"].values[:, 1, :], 2.0)

    @pytest.mark.unit
    @patch("aef_loader.utils.xr_reproject")
    def test_overlap_first_zone_takes_precedence(self, mock_reproject):
        """In overlapping regions (both non-NaN), zone 1 values should be kept."""
        band_names = ["A00"]

        arr1 = np.array([[[1.0, 1.0]]], dtype=np.float32)
        arr2 = np.array([[[2.0, 2.0]]], dtype=np.float32)

        ds1 = xr.Dataset(
            {"embeddings": (["band", "y", "x"], da.from_array(arr1, chunks=(1, 1, 2)))},
            coords={"band": band_names, "y": [0.0], "x": [0.0, 1.0]},
        )
        ds2 = xr.Dataset(
            {"embeddings": (["band", "y", "x"], da.from_array(arr2, chunks=(1, 1, 2)))},
            coords={"band": band_names, "y": [0.0], "x": [0.0, 1.0]},
        )

        mock_reproject.side_effect = [ds1, ds2]
        tree, geobox = _make_datatree_and_geobox(
            [_make_dask_dataset((1, 1, 2), band_names)] * 2
        )

        result = reproject_datatree(tree, geobox).compute()

        # Zone 1 values should win (fillna only fills where combined is NaN)
        np.testing.assert_array_equal(result["embeddings"].values, 1.0)

    @pytest.mark.unit
    @patch("aef_loader.utils.xr_reproject")
    def test_three_zones_merge(self, mock_reproject):
        """Three zones should merge correctly with fillna semantics."""
        band_names = ["A00"]

        # Zone 1: col 0 valid, cols 1-2 NaN
        arr1 = np.array([[[1.0, np.nan, np.nan]]], dtype=np.float32)
        # Zone 2: col 1 valid, cols 0,2 NaN
        arr2 = np.array([[[np.nan, 2.0, np.nan]]], dtype=np.float32)
        # Zone 3: col 2 valid, cols 0-1 NaN
        arr3 = np.array([[[np.nan, np.nan, 3.0]]], dtype=np.float32)

        ds1 = xr.Dataset(
            {"embeddings": (["band", "y", "x"], da.from_array(arr1, chunks=(1, 1, 3)))},
            coords={"band": band_names, "y": [0.0], "x": [0.0, 1.0, 2.0]},
        )
        ds2 = xr.Dataset(
            {"embeddings": (["band", "y", "x"], da.from_array(arr2, chunks=(1, 1, 3)))},
            coords={"band": band_names, "y": [0.0], "x": [0.0, 1.0, 2.0]},
        )
        ds3 = xr.Dataset(
            {"embeddings": (["band", "y", "x"], da.from_array(arr3, chunks=(1, 1, 3)))},
            coords={"band": band_names, "y": [0.0], "x": [0.0, 1.0, 2.0]},
        )

        mock_reproject.side_effect = [ds1, ds2, ds3]
        tree, geobox = _make_datatree_and_geobox(
            [_make_dask_dataset((1, 1, 3), band_names)] * 3
        )

        result = reproject_datatree(tree, geobox).compute()

        np.testing.assert_array_equal(
            result["embeddings"].values[0, 0, :], [1.0, 2.0, 3.0]
        )

    @pytest.mark.unit
    @patch("aef_loader.utils.xr_reproject")
    def test_shape_mismatch_raises(self, mock_reproject):
        """Mismatched shapes between zones should raise AssertionError."""
        band_names = ["A00"]

        ds1 = _make_dask_dataset((1, 10, 10), band_names, fill_value=1.0)
        ds2 = _make_dask_dataset((1, 10, 20), band_names, fill_value=2.0)

        mock_reproject.side_effect = [ds1, ds2]
        tree, geobox = _make_datatree_and_geobox(
            [_make_dask_dataset((1, 10, 10), band_names),
             _make_dask_dataset((1, 10, 20), band_names)]
        )

        with pytest.raises(AssertionError, match="Shape mismatch"):
            reproject_datatree(tree, geobox)

    @pytest.mark.unit
    def test_empty_tree_raises(self):
        """Empty DataTree should raise ValueError."""
        tree = DataTree()
        geobox = MagicMock()

        with pytest.raises(ValueError, match="No datasets to reproject"):
            reproject_datatree(tree, geobox)

    @pytest.mark.unit
    @patch("aef_loader.utils.xr_reproject")
    def test_attrs_preserved(self, mock_reproject):
        """Output should contain source_zones and target_crs attributes."""
        band_names = ["A00"]
        ds1 = _make_dask_dataset((1, 5, 5), band_names, fill_value=1.0)
        ds2 = _make_dask_dataset((1, 5, 5), band_names, fill_value=2.0)

        mock_reproject.side_effect = [ds1, ds2]
        tree, geobox = _make_datatree_and_geobox(
            [ds1.copy(), ds2.copy()],
            zone_names=["10N", "11N"],
        )

        result = reproject_datatree(tree, geobox)

        assert "source_zones" in result.attrs
        assert result.attrs["target_crs"] == "EPSG:4326"

    @pytest.mark.unit
    @patch("aef_loader.utils.xr_reproject")
    def test_multiple_data_vars(self, mock_reproject):
        """Merge should work across multiple data variables."""
        band_names = ["A00"]
        shape = (1, 5, 5)
        chunks = (1, 5, 5)

        arr1_a = da.full(shape, 1.0, dtype=np.float32, chunks=chunks)
        arr1_b = da.full(shape, np.nan, dtype=np.float32, chunks=chunks)
        arr2_a = da.full(shape, np.nan, dtype=np.float32, chunks=chunks)
        arr2_b = da.full(shape, 2.0, dtype=np.float32, chunks=chunks)

        coords = {"band": band_names, "y": np.arange(5.0), "x": np.arange(5.0)}
        ds1 = xr.Dataset(
            {
                "embeddings": (["band", "y", "x"], arr1_a),
                "quality": (["band", "y", "x"], arr1_b),
            },
            coords=coords,
        )
        ds2 = xr.Dataset(
            {
                "embeddings": (["band", "y", "x"], arr2_a),
                "quality": (["band", "y", "x"], arr2_b),
            },
            coords=coords,
        )

        mock_reproject.side_effect = [ds1, ds2]
        tree, geobox = _make_datatree_and_geobox(
            [_make_dask_dataset((1, 5, 5), band_names)] * 2
        )

        result = reproject_datatree(tree, geobox).compute()

        np.testing.assert_array_equal(result["embeddings"].values, 1.0)
        np.testing.assert_array_equal(result["quality"].values, 2.0)

    @pytest.mark.unit
    @patch("aef_loader.utils.xr_reproject")
    def test_no_rechunk_merge_tasks_in_graph(self, mock_reproject):
        """The dask graph should not contain rechunk-merge tasks."""
        n_bands = 64
        band_names = [f"A{i:02d}" for i in range(n_bands)]
        shape = (n_bands, 20, 20)
        chunks = (n_bands, 20, 20)

        arr1 = da.full(shape, 1.0, dtype=np.float32, chunks=chunks)
        arr2 = da.full(shape, 2.0, dtype=np.float32, chunks=chunks)

        coords = {
            "band": band_names,
            "y": np.arange(20.0),
            "x": np.arange(20.0),
        }
        ds1 = xr.Dataset({"embeddings": (["band", "y", "x"], arr1)}, coords=coords)
        ds2 = xr.Dataset({"embeddings": (["band", "y", "x"], arr2)}, coords=coords)

        mock_reproject.side_effect = [ds1, ds2]
        tree, geobox = _make_datatree_and_geobox(
            [_make_dask_dataset(shape, band_names)] * 2
        )

        result = reproject_datatree(tree, geobox)

        # Inspect dask graph keys for rechunk-merge tasks
        graph_keys = set()
        for var in result.data_vars:
            graph_dict = dict(result[var].data.__dask_graph__())
            for key in graph_dict:
                if isinstance(key, tuple):
                    graph_keys.add(key[0])
                else:
                    graph_keys.add(str(key))

        rechunk_keys = [k for k in graph_keys if "rechunk-merge" in str(k)]
        assert len(rechunk_keys) == 0, (
            f"Found rechunk-merge tasks in graph: {rechunk_keys}"
        )


def _delayed_zone_data(n_bands, ny, nx, left=True):
    """Create a dask delayed object for zone data."""
    import dask

    @dask.delayed
    def _make():
        arr = np.full((n_bands, ny, nx), np.nan, dtype=np.float32)
        half = nx // 2
        if left:
            arr[:, :, :half] = 1.0
        else:
            arr[:, :, half:] = 2.0
        return arr

    return _make()
