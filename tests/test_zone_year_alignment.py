"""reproject_datatree must merge zones that hold different sets of years."""

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import assign_crs
from xarray import DataTree

from aef_loader.utils import reproject_datatree

CRS = "EPSG:32631"
SOURCE = GeoBox.from_bbox((499000, 3994000, 501000, 3996000), crs=CRS, resolution=20)
TARGET = GeoBox.from_bbox((3.0, 36.1, 3.01, 36.12), crs="EPSG:4326", resolution=0.0005)


def _zone(years, value):
    height, width = SOURCE.shape
    emb = xr.DataArray(
        np.full((len(years), 2, height, width), value, np.int8),
        dims=("time", "band", "y", "x"),
        coords={
            "time": years,
            "band": ["A00", "A01"],
            "y": SOURCE.coords["y"].values,
            "x": SOURCE.coords["x"].values,
        },
        attrs={"nodata": -128, "_FillValue": -128},
    )
    return assign_crs(xr.Dataset({"embeddings": emb}), CRS)


@pytest.mark.unit
def test_zones_with_different_years_merge():
    tree = DataTree.from_dict(
        {"31N": DataTree(_zone([2023], 5)), "32N": DataTree(_zone([2024], 9))}
    )

    out = reproject_datatree(tree, TARGET).compute()

    assert list(out.time.values) == [2023, 2024]
    assert out.embeddings.dtype == np.int8
    assert out.embeddings.attrs["nodata"] == -128
    covered = out.embeddings.sel(time=2023).values
    assert set(np.unique(covered)) == {-128, 5}
    assert set(np.unique(out.embeddings.sel(time=2024).values)) == {-128, 9}


@pytest.mark.unit
def test_zones_with_same_years_unchanged():
    tree = DataTree.from_dict(
        {"31N": DataTree(_zone([2024], 5)), "32N": DataTree(_zone([2024], 9))}
    )

    out = reproject_datatree(tree, TARGET).compute()

    assert list(out.time.values) == [2024]
    assert set(np.unique(out.embeddings.values)) == {-128, 5}
