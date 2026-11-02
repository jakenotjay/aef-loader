"""Tests for aef_loader.reader module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from aef_loader.reader import (
    VirtualTiffReader,
    _get_geobox_from_dataset,
    _get_overview_geobox,
    _parse_gcs_path,
)


class TestParseGcsPath:
    """Tests for _parse_gcs_path helper."""

    @pytest.mark.unit
    def test_parse_gs_url(self):
        """Test parsing gs:// URL."""
        bucket, key = _parse_gcs_path("gs://my-bucket/path/to/file.tif")

        assert bucket == "my-bucket"
        assert key == "path/to/file.tif"


class TestVirtualTiffReader:
    """Tests for VirtualTiffReader class."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with VirtualTiffReader() as reader:
            assert reader is not None

        # After exit, stores should be cleared
        assert reader._stores == {}

    @pytest.mark.unit
    def test_get_gcs_store_with_project(self):
        """Test that GCS store includes requester-pays header when project is set."""
        reader = VirtualTiffReader(gcp_project="my-project")
        with patch("aef_loader.reader.GCSStore") as mock_gcs:
            reader._get_gcs_store("my-bucket")
            mock_gcs.assert_called_once_with(
                bucket="my-bucket",
                client_options={
                    "default_headers": {"x-goog-user-project": "my-project"}
                },
            )

    @pytest.mark.unit
    def test_get_gcs_store_without_project_raises(self):
        """Test that GCS store raises when no project is set."""
        reader = VirtualTiffReader()
        with pytest.raises(ValueError, match="gcp_project is required"):
            reader._get_gcs_store("my-bucket")

    @pytest.mark.unit
    def test_get_store_caches(self):
        """Test that _get_store caches stores."""
        reader = VirtualTiffReader()
        mock_store = MagicMock()
        reader._stores["gs://my-bucket"] = mock_store

        with patch("obstore.store.GCSStore") as mock_gcs:
            store = reader._get_store("gs", "my-bucket")

            mock_gcs.assert_not_called()
            assert store == mock_store


def _make_ds(height: int, width: int, attrs: dict) -> xr.Dataset:
    da = xr.DataArray(
        np.zeros((1, height, width), dtype="int8"),
        dims=("band", "y", "x"),
        attrs=attrs,
    )
    return xr.Dataset({"0": da})


class TestOverviewGeobox:
    """Overview IFDs carry no georeferencing tags; derive it from IFD 0."""

    @pytest.mark.unit
    def test_overview_from_pixel_scale(self):
        ds0 = _make_ds(
            8192,
            8192,
            {
                "model_pixel_scale": (10.0, 10.0, 0.0),
                "model_tiepoint": (0, 0, 0, 500000.0, 100000.0, 0),
            },
        )
        ds_ovr = _make_ds(1024, 1024, {})

        gbox = _get_overview_geobox(ds_ovr, ds0, "EPSG:32647")

        assert gbox.shape == (1024, 1024)
        assert gbox.affine.a == pytest.approx(80.0)
        assert gbox.affine.e == pytest.approx(80.0)
        assert (gbox.affine.c, gbox.affine.f) == (500000.0, 100000.0)

    @pytest.mark.unit
    def test_overview_from_model_transformation_keeps_sign(self):
        # bottom-up image: positive y scale
        transform = (
            10.0, 0, 0, 500000.0,
            0, -10.0, 0, 200000.0,
            0, 0, 0, 0,
            0, 0, 0, 1,
        )  # fmt: skip
        ds0 = _make_ds(4096, 8192, {"model_transformation": transform})
        ds_ovr = _make_ds(1024, 1024, {})

        gbox = _get_overview_geobox(ds_ovr, ds0, "EPSG:32647")

        assert gbox.affine.a == pytest.approx(80.0)
        assert gbox.affine.e == pytest.approx(-40.0)
        assert (gbox.affine.c, gbox.affine.f) == (500000.0, 200000.0)

    @pytest.mark.unit
    def test_same_dataset_unchanged(self):
        attrs = {
            "model_pixel_scale": (10.0, 10.0, 0.0),
            "model_tiepoint": (0, 0, 0, 500000.0, 100000.0, 0),
        }
        ds0 = _make_ds(64, 64, attrs)

        assert (
            _get_overview_geobox(ds0, ds0, "EPSG:32647").affine
            == _get_geobox_from_dataset(ds0, "EPSG:32647").affine
        )
