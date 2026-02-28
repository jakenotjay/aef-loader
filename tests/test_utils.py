"""Tests for aef_loader.utils module."""

import numpy as np
import pytest
import xarray as xr
from aef_loader.utils import (
    dequantize_aef,
    mask_nodata,
    quantize_aef,
    split_bands,
)


class TestDequantizeAef:
    """Tests for dequantize_aef function.

    The correct formula is: ((value / 127.5) ** 2) * sign(value)
    This maps int8 [-127, 127] to float32 [-1, 1].
    """

    @pytest.mark.unit
    def test_dequantize_numpy_array(self):
        """Test dequantization formula across positive, negative, zero, and nodata values."""
        quantized = np.array([127, -127, 0, 64, -64, -128], dtype=np.int8)
        result = dequantize_aef(quantized)

        expected = np.array(
            [
                ((127 / 127.5) ** 2) * 1,
                ((-127 / 127.5) ** 2) * -1,
                0.0,
                ((64 / 127.5) ** 2) * 1,
                ((-64 / 127.5) ** 2) * -1,
                np.nan,  # nodata
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_almost_equal(result[:5], expected[:5], decimal=4)
        assert np.isnan(result[5])

    @pytest.mark.unit
    def test_dequantize_xarray_dataarray(self):
        """Test dequantization of xarray DataArray sets expected attrs."""
        data = np.array([127, -127, 0], dtype=np.int8)
        da = xr.DataArray(data, dims=["x"])

        result = dequantize_aef(da)

        assert result.dtype == np.float32
        assert result.attrs["dequantized"] is True

    @pytest.mark.unit
    def test_dequantize_xarray_dataset(self):
        """Test dequantization of xarray Dataset maps over all variables."""
        data = np.array([[127, -127], [64, 0]], dtype=np.int8)
        ds = xr.Dataset({"var1": (["y", "x"], data)})

        result = dequantize_aef(ds)

        assert result["var1"].dtype == np.float32


class TestQuantizeAef:
    """Tests for quantize_aef function.

    The quantization formula is: sign(v) * sqrt(|v|) * 127.5
    This is the inverse of dequantization: ((v / 127.5) ** 2) * sign(v)
    """

    @pytest.mark.unit
    def test_quantize_boundary_values(self):
        """Test quantization of 1.0 -> 127 and -1.0 -> -127."""
        data = np.array([1.0, -1.0, 0.0], dtype=np.float32)
        result = quantize_aef(data)

        np.testing.assert_array_equal(result, np.array([127, -127, 0], dtype=np.int8))

    @pytest.mark.unit
    def test_quantize_clamps_to_valid_range(self):
        """Test that values outside [-1, 1] clamp to [-127, 127] without int8 overflow."""
        data = np.array([2.0, -2.0], dtype=np.float32)
        result = quantize_aef(data)

        assert result.max() <= 127
        assert result.min() >= -127

    @pytest.mark.unit
    def test_roundtrip(self):
        """Test that quantize -> dequantize is approximately identity."""
        original = np.array([1.0, -1.0, 0.5, -0.5, 0.0, 0.25], dtype=np.float32)
        quantized = quantize_aef(original)
        dequantized = dequantize_aef(quantized)

        np.testing.assert_allclose(original, dequantized, rtol=0.02, atol=0.01)


class TestSplitBands:
    """Tests for split_bands function."""

    @pytest.mark.unit
    def test_split_bands_basic(self):
        """Test splitting embeddings variable into per-band variables."""
        data = np.arange(4 * 5 * 5, dtype=np.int8).reshape(4, 5, 5)
        band_names = [f"A{i:02d}" for i in range(4)]
        da = xr.DataArray(data, dims=["band", "y", "x"], coords={"band": band_names})
        da.name = "embeddings"
        ds = da.to_dataset()
        ds.attrs["test_attr"] = "preserved"

        result = split_bands(ds)

        assert len(result.data_vars) == 4
        assert "A00" in result.data_vars
        assert "A03" in result.data_vars
        assert "embeddings" not in result.data_vars
        assert result.attrs["test_attr"] == "preserved"
        np.testing.assert_array_equal(result["A00"].values, data[0])
        np.testing.assert_array_equal(result["A03"].values, data[3])


class TestMaskNodata:
    """Tests for mask_nodata function."""

    @pytest.mark.unit
    def test_mask_nodata_numpy(self):
        """Test NoData masking for numpy array."""
        data = np.array([127, -128, 0, -128, 64], dtype=np.int8)
        result = mask_nodata(data)

        assert np.isnan(result[1])
        assert np.isnan(result[3])
        assert not np.isnan(result[0])
        assert not np.isnan(result[2])
        assert not np.isnan(result[4])

    @pytest.mark.unit
    def test_mask_nodata_xarray(self):
        """Test NoData masking for xarray DataArray."""
        data = np.array([127, -128, 0], dtype=np.int8)
        da = xr.DataArray(data, dims=["x"])

        result = mask_nodata(da)

        assert result.isnull()[1].item()
        assert not result.isnull()[0].item()
        assert not result.isnull()[2].item()
