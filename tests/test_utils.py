"""Tests for aef_loader.utils module."""

import numpy as np
import pytest
import xarray as xr
from aef_loader.utils import (
    bbox_to_utm_zone,
    compute_embedding_similarity,
    dequantize_aef,
    get_channel_names,
    mask_nodata,
    normalize_embeddings,
    quantize_aef,
    split_bands,
    transform_bbox,
    wgs84_to_utm,
)


class TestDequantizeAef:
    """Tests for dequantize_aef function.

    The correct formula is: ((value / 127.5) ** 2) * sign(value)
    This maps int8 [-127, 127] to float32 [-1, 1].
    """

    @pytest.mark.unit
    def test_dequantize_max_positive(self):
        """Test dequantization of max positive value (127 -> ~1.0)."""
        quantized = np.array([127], dtype=np.int8)
        result = dequantize_aef(quantized)

        # ((127 / 127.5) ** 2) * 1 ≈ 0.9922
        expected = ((127 / 127.5) ** 2) * 1
        np.testing.assert_array_almost_equal(result, [expected], decimal=4)

    @pytest.mark.unit
    def test_dequantize_max_negative(self):
        """Test dequantization of max negative value (-127 -> ~-1.0)."""
        quantized = np.array([-127], dtype=np.int8)
        result = dequantize_aef(quantized)

        # ((-127 / 127.5) ** 2) * -1 ≈ -0.9922
        expected = ((-127 / 127.5) ** 2) * -1
        np.testing.assert_array_almost_equal(result, [expected], decimal=4)

    @pytest.mark.unit
    def test_dequantize_zero(self):
        """Test dequantization of zero."""
        quantized = np.array([0], dtype=np.int8)
        result = dequantize_aef(quantized)

        np.testing.assert_array_almost_equal(result, [0.0], decimal=4)

    @pytest.mark.unit
    def test_dequantize_intermediate_value(self):
        """Test dequantization of intermediate value."""
        quantized = np.array([64], dtype=np.int8)
        result = dequantize_aef(quantized)

        # ((64 / 127.5) ** 2) * 1 ≈ 0.252
        expected = ((64 / 127.5) ** 2) * 1
        np.testing.assert_array_almost_equal(result, [expected], decimal=4)

    @pytest.mark.unit
    def test_dequantize_numpy_array(self):
        """Test dequantization of numpy array."""
        quantized = np.array([127, -127, 0, 64, -64], dtype=np.int8)
        result = dequantize_aef(quantized)

        expected = np.array(
            [
                ((127 / 127.5) ** 2) * 1,
                ((-127 / 127.5) ** 2) * -1,
                0.0,
                ((64 / 127.5) ** 2) * 1,
                ((-64 / 127.5) ** 2) * -1,
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    @pytest.mark.unit
    def test_dequantize_xarray_dataarray(self):
        """Test dequantization of xarray DataArray."""
        data = np.array([127, -127, 0], dtype=np.int8)
        da = xr.DataArray(data, dims=["x"])

        result = dequantize_aef(da)

        assert result.dtype == np.float32
        assert result.attrs["dequantized"] is True

    @pytest.mark.unit
    def test_dequantize_xarray_dataset(self):
        """Test dequantization of xarray Dataset."""
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
    def test_quantize_one(self):
        """Test quantization of 1.0 -> 127."""
        data = np.array([1.0], dtype=np.float32)
        result = quantize_aef(data)

        # sign(1) * sqrt(1) * 127.5 = 127.5 -> rounds to 127
        expected = np.array([127], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    def test_quantize_negative_one(self):
        """Test quantization of -1.0 -> -127."""
        data = np.array([-1.0], dtype=np.float32)
        result = quantize_aef(data)

        expected = np.array([-127], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    def test_quantize_zero(self):
        """Test quantization of 0.0 -> 0."""
        data = np.array([0.0], dtype=np.float32)
        result = quantize_aef(data)

        expected = np.array([0], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    def test_quantize_clamps_to_valid_range(self):
        """Test that quantization clamps values to [-127, 127]."""
        # Values > 1.0 should still clamp to 127
        data = np.array([2.0, -2.0], dtype=np.float32)
        result = quantize_aef(data)

        assert result.max() <= 127
        assert result.min() >= -127

    @pytest.mark.unit
    def test_roundtrip_quantize_dequantize(self):
        """Test that quantize -> dequantize is approximately identity."""
        original = np.array([1.0, -1.0, 0.5, -0.5, 0.0, 0.25], dtype=np.float32)
        quantized = quantize_aef(original)
        dequantized = dequantize_aef(quantized)

        # Allow for some quantization error (about 2%)
        np.testing.assert_allclose(original, dequantized, rtol=0.02, atol=0.01)


class TestGetChannelNames:
    """Tests for get_channel_names function."""

    @pytest.mark.unit
    def test_default_channel_names(self):
        """Test default 64 channel names (A00-A63)."""
        names = get_channel_names()

        assert len(names) == 64
        assert names[0] == "A00"
        assert names[63] == "A63"
        assert names[10] == "A10"

    @pytest.mark.unit
    def test_custom_num_channels(self):
        """Test custom number of channels."""
        names = get_channel_names(num_channels=4)

        assert len(names) == 4
        assert names == ["A00", "A01", "A02", "A03"]

    @pytest.mark.unit
    def test_custom_prefix(self):
        """Test custom prefix."""
        names = get_channel_names(num_channels=3, prefix="B")

        assert names == ["B00", "B01", "B02"]


class TestSplitBands:
    """Tests for split_bands function."""

    @pytest.mark.unit
    def test_split_bands_basic(self):
        """Test splitting embeddings variable into A00-A63 variables."""
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

    @pytest.mark.unit
    def test_split_bands_64_channels(self):
        """Test splitting 64 bands produces A00-A63."""
        data = np.zeros((64, 3, 3), dtype=np.int8)
        band_names = [f"A{i:02d}" for i in range(64)]
        da = xr.DataArray(data, dims=["band", "y", "x"], coords={"band": band_names})
        da.name = "embeddings"
        ds = da.to_dataset()

        result = split_bands(ds)

        assert len(result.data_vars) == 64
        assert list(result.data_vars) == band_names

    @pytest.mark.unit
    def test_split_bands_custom_var_name(self):
        """Test splitting with a custom variable name."""
        data = np.zeros((2, 3, 3), dtype=np.int8)
        band_names = ["A00", "A01"]
        da = xr.DataArray(data, dims=["band", "y", "x"], coords={"band": band_names})
        da.name = "my_var"
        ds = da.to_dataset()

        result = split_bands(ds, var="my_var")

        assert len(result.data_vars) == 2
        assert "A00" in result.data_vars
        assert "A01" in result.data_vars


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

    @pytest.mark.unit
    def test_mask_nodata_custom_value(self):
        """Test NoData masking with custom nodata value."""
        data = np.array([127, -100, 0], dtype=np.int8)
        result = mask_nodata(data, nodata_value=-100)

        assert np.isnan(result[1])
        assert not np.isnan(result[0])


class TestNormalizeEmbeddings:
    """Tests for normalize_embeddings function."""

    @pytest.mark.unit
    def test_normalize_numpy(self):
        """Test L2 normalization of numpy array."""
        data = np.array([[[3, 0], [0, 4]]], dtype=np.float32)  # Shape: (1, 2, 2)
        result = normalize_embeddings(data)

        # Expected: divide by norm along channel axis
        # [3,0,0,4] norms are [3,0,0,4], normalized: [1,0,0,1]
        expected = np.array([[[1, 0], [0, 1]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    @pytest.mark.unit
    def test_normalize_handles_zero(self):
        """Test that zero vectors don't cause division by zero."""
        data = np.array([[[0, 1], [0, 0]]], dtype=np.float32)
        result = normalize_embeddings(data)

        # Should not have NaN values
        assert not np.isnan(result).any()


class TestComputeEmbeddingSimilarity:
    """Tests for compute_embedding_similarity function."""

    @pytest.mark.unit
    def test_identical_embeddings(self):
        """Test similarity of identical embeddings."""
        emb = np.random.randn(64, 10, 10).astype(np.float32)
        result = compute_embedding_similarity(emb, emb)

        # Identical embeddings should have similarity ~1
        np.testing.assert_array_almost_equal(result, np.ones((10, 10)), decimal=5)

    @pytest.mark.unit
    def test_orthogonal_embeddings(self):
        """Test similarity of orthogonal embeddings."""
        emb1 = np.zeros((2, 1, 1), dtype=np.float32)
        emb1[0] = 1
        emb2 = np.zeros((2, 1, 1), dtype=np.float32)
        emb2[1] = 1

        result = compute_embedding_similarity(emb1, emb2)

        # Orthogonal embeddings should have similarity ~0
        np.testing.assert_array_almost_equal(result, np.zeros((1, 1)), decimal=5)


class TestWgs84ToUtm:
    """Tests for wgs84_to_utm function."""

    @pytest.mark.unit
    def test_northern_hemisphere(self):
        """Test UTM zone for northern hemisphere."""
        # San Francisco: lon=-122, lat=37 -> UTM zone 10N (EPSG:32610)
        epsg = wgs84_to_utm(-122.0, 37.0)
        assert epsg == 32610

    @pytest.mark.unit
    def test_southern_hemisphere(self):
        """Test UTM zone for southern hemisphere."""
        # Sydney: lon=151, lat=-34 -> UTM zone 56S (EPSG:32756)
        epsg = wgs84_to_utm(151.0, -34.0)
        assert epsg == 32756


class TestBboxToUtmZone:
    """Tests for bbox_to_utm_zone function."""

    @pytest.mark.unit
    def test_bbox_to_utm(self):
        """Test UTM zone from bbox."""
        bbox = (-122.5, 37.5, -121.5, 38.5)
        epsg = bbox_to_utm_zone(bbox)
        assert epsg == 32610


class TestTransformBbox:
    """Tests for transform_bbox function."""

    @pytest.mark.unit
    def test_transform_wgs84_to_utm(self):
        """Test bbox transformation from WGS84 to UTM."""
        bbox = (-122.0, 37.0, -121.0, 38.0)
        result = transform_bbox(bbox, 4326, 32610)

        # Result should be in meters (UTM)
        assert result[0] > 100000  # Easting should be large
        assert result[2] > 100000
        assert result[1] > 4000000  # Northing should be large
        assert result[3] > 4000000
