"""Tests for aef_loader._cloud helpers."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from aef_loader._cloud import (
    _affine_from_pixel_scale_and_tiepoint,
    get_geobox_from_dataset,
)


class TestAffineFromPixelScaleAndTiepoint:
    @pytest.mark.unit
    def test_affine_from_pixel_scale_implies_negative_y_step(self):
        """GeoTIFF spec: ModelPixelScaleY is positive but the affine y-step is
        implicitly negative (north-up, NW-corner origin)."""
        affine = _affine_from_pixel_scale_and_tiepoint(
            (0.0001, 0.0001, 0.0),
            (0, 0, 0, 9.0, 6.0, 0),
        )
        assert affine.a == 0.0001
        assert affine.b == 0
        assert affine.c == 9.0
        assert affine.d == 0
        assert affine.e == -0.0001
        assert affine.f == 6.0


class TestGetGeoboxFromDataset:
    @pytest.mark.unit
    def test_geobox_north_up_for_pixel_scale_path(self):
        """End-to-end: ds with model_pixel_scale + model_tiepoint produces a
        north-up geobox whose top bound matches the tiepoint y."""
        # NOTE: a data_var literally named "x" gets promoted to a coordinate
        # by xarray (matches the dim name), so we use "band" here. The shape
        # of the underlying array is what matters for the geobox extraction.
        ds = xr.Dataset(
            data_vars={"band": (("y", "x"), np.zeros((4, 4)))},
        )
        ds["band"].attrs["model_pixel_scale"] = [0.0001, 0.0001, 0.0]
        ds["band"].attrs["model_tiepoint"] = [0, 0, 0, 9.0, 6.0, 0]

        geobox = get_geobox_from_dataset(ds, "EPSG:4326")

        assert geobox.affine.e < 0  # north-up
        bbox = geobox.boundingbox
        assert bbox.top == pytest.approx(6.0)
        assert bbox.bottom == pytest.approx(6.0 - 4 * 0.0001)
