"""Benchmark 1: Load a single COG tile, download the entire image (all 64 bands)
and saves it to zarr.

Compares aef-loader (virtual-tiff) against:
- rioxarray
- xee
- rasteret

Run:
    uv run pytest benchmarks/test_single_tile.py -v
"""
# TODO: update the number of rounds to 3, once finished developing

from __future__ import annotations

import asyncio
import os
import shutil

import pytest

from aef_loader.reader import VirtualTiffReader


@pytest.mark.benchmark(group="single-tile")
def test_aef_loader(benchmark, single_tile, tmp_path):
    """Load one tile via VirtualTiffReader, compute all bands, write to zarr."""
    zarr_path = tmp_path / "output.zarr"

    def _setup():
        if zarr_path.exists():
            shutil.rmtree(zarr_path)

    def _load():
        async def _inner():
            async with VirtualTiffReader() as reader:
                tree = await reader.open_tiles_by_zone(single_tile)
            zone = list(tree.children.keys())[0]
            return tree[zone].ds

        ds = asyncio.run(_inner())
        ds.to_zarr(zarr_path)
        return ds

    result = benchmark.pedantic(
        _load, setup=_setup, rounds=1, iterations=1, warmup_rounds=0
    )
    assert result["embeddings"].sizes["band"] == 64
    assert result["embeddings"].sizes["y"] > 0
    assert result["embeddings"].sizes["x"] > 0


@pytest.mark.benchmark(group="single-tile")
def test_rioxarray(benchmark, single_tile_url, tmp_path):
    """Load one tile via rioxarray.open_rasterio, compute all bands, write to zarr."""
    rioxarray = pytest.importorskip("rioxarray")
    import rioxarray  # noqa: F811 — needed for the .rio accessor

    # AWS_NO_SIGN_REQUEST lets GDAL read public S3 without credentials
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    zarr_path = tmp_path / "output.zarr"

    def _setup():
        if zarr_path.exists():
            shutil.rmtree(zarr_path)

    def _load():
        da = rioxarray.open_rasterio(single_tile_url, chunks="auto")  # type: ignore[arg-type]
        da.to_dataset(name="embeddings").to_zarr(zarr_path)  # type: ignore[union-attr]
        return da

    result = benchmark.pedantic(
        _load, setup=_setup, rounds=1, iterations=1, warmup_rounds=0
    )
    assert result.shape[0] == 64


@pytest.mark.skip(
    reason="xee not yet installed; requires GCP credentials + Earth Engine auth"
)
@pytest.mark.benchmark(group="single-tile")
def test_xee(benchmark, gcp_project):
    """Load one tile via xee (Google Earth Engine + xarray).

    Intended usage (uncomment when xee is available):

        import xarray as xr
        import ee

        ee.Initialize(project=gcp_project)
        ds = xr.open_dataset(
            "ee://projects/alphaearth/assets/satellite_embedding/v1",
            engine="ee",
            geometry=ee.Geometry.Rectangle(list(BENCH_BBOX)),
            scale=10,
        )
        result = ds.isel(band=0, x=slice(0, 256), y=slice(0, 256)).compute()
    """
    pass


# ---------------------------------------------------------------------------
# rasteret (skeleton)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="rasteret not yet installed")
@pytest.mark.benchmark(group="single-tile")
def test_rasteret(benchmark, single_tile_url):
    """Load one tile via rasteret.

    Intended usage (uncomment when rasteret is available):

        from rasteret import open_raster
        ds = open_raster(single_tile_url)
        result = ds.isel(band=0, y=slice(0, 256), x=slice(0, 256)).compute()
    """
    pass
