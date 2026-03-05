"""Slow tests verifying nodata attrs survive load -> reproject -> combine.

Uses Source Cooperative (public, no credentials needed) with a bbox
spanning two UTM zones to exercise xr_reproject and combine_first.

Run manually:
    uv run pytest -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from aef_loader.constants import AEF_NODATA_VALUE, DataSource
from aef_loader.index import AEFIndex
from aef_loader.reader import VirtualTiffReader
from aef_loader.utils import reproject_datatree

# Bbox spanning UTM zones 36S/37S boundary in East Africa
BBOX = (35.8, -6.5, 36.2, -6.0)
YEAR = 2023


def assert_nodata_attrs(ds, expected_nodata=AEF_NODATA_VALUE):
    """Assert both nodata and _FillValue are set correctly on all data vars."""
    for var in ds.data_vars:
        attrs = ds[var].attrs
        nodata = attrs.get("nodata")
        fill = attrs.get("_FillValue")

        if isinstance(expected_nodata, float) and np.isnan(expected_nodata):
            assert np.isnan(nodata), f"{var}: nodata={nodata}, expected NaN"
            assert np.isnan(fill), f"{var}: _FillValue={fill}, expected NaN"
        else:
            assert nodata == expected_nodata, (
                f"{var}: nodata={nodata}, expected {expected_nodata}"
            )
            assert fill == expected_nodata, (
                f"{var}: _FillValue={fill}, expected {expected_nodata}"
            )


@pytest.fixture(scope="module")
def source_coop_tree():
    """Load tiles spanning two UTM zones from Source Cooperative."""
    import asyncio

    async def _load():
        index = AEFIndex(source=DataSource.SOURCE_COOP)
        index.load()
        tiles = await index.query(bbox=BBOX, years=YEAR, limit=4)
        async with VirtualTiffReader() as reader:
            return await reader.open_tiles_by_zone(tiles), tiles

    tree, tiles = asyncio.run(_load())
    zones = {t.utm_zone for t in tiles}
    assert len(zones) >= 2, f"Expected tiles in >=2 zones, got {zones}"
    return tree


@pytest.mark.slow
class TestNodataReproject:
    """Verify nodata=-128 survives the full reproject pipeline."""

    def test_raw_loaded_attrs(self, source_coop_tree):
        """Raw loaded data has nodata=-128 and _FillValue=-128 on all zones."""
        for zone_name in source_coop_tree.children:
            ds = source_coop_tree[zone_name].ds
            if ds is None or len(ds.data_vars) == 0:
                continue
            assert_nodata_attrs(ds, AEF_NODATA_VALUE)

    def test_reproject_explicit_dst_nodata(self, source_coop_tree):
        """Reprojected data with explicit dst_nodata keeps nodata=-128."""
        from odc.geo.geobox import GeoBox

        target = GeoBox.from_bbox(bbox=BBOX, crs="EPSG:4326", resolution=0.001)
        combined = reproject_datatree(
            source_coop_tree, target, dst_nodata=AEF_NODATA_VALUE
        )
        assert_nodata_attrs(combined, AEF_NODATA_VALUE)

    def test_reproject_nodata_from_attrs(self, source_coop_tree):
        """Reprojected data with dst_nodata=None picks up nodata from attrs."""
        from odc.geo.geobox import GeoBox

        target = GeoBox.from_bbox(bbox=BBOX, crs="EPSG:4326", resolution=0.001)
        combined = reproject_datatree(source_coop_tree, target)
        assert_nodata_attrs(combined, AEF_NODATA_VALUE)
