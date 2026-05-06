"""Reader for Forest Data Partnership commodity-probability COGs.

Sibling to :class:`aef_loader.reader.VirtualTiffReader` for AEF embeddings.
The two readers intentionally duplicate their obstore/registry bookkeeping
rather than share a base class — the data shapes diverge (single-band float32
EPSG:4326 mosaic vs 64-band int8 UTM-zoned embeddings) and a third dataset
would be the right time to factor.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections import defaultdict
from typing import Literal

import numpy as np
import xarray as xr
from odc.geo.xr import assign_crs, xr_coords
from virtual_tiff import VirtualTIFF
from obspec_utils.registry import ObjectStoreRegistry
from xarray import DataTree

from aef_loader._cloud import (
    PathProtocol,
    get_geobox_from_dataset,
    make_gcs_store,
    parse_cloud_path,
)
from aef_loader.types import FDPTileInfo
from aef_loader.utils import set_aef_nodata

logger = logging.getLogger(__name__)

# FDP tiles are always EPSG:4326. Hardcoded (rather than read from
# tile.crs_epsg) so the orientation expectations of get_geobox_from_dataset
# stay obvious at the call site.
_FDP_CRS = "EPSG:4326"


class FDPReader:
    """Open FDP commodity-probability COGs as a per-commodity DataTree.

    Each input tile is a 1°×1° float32 EPSG:4326 COG with a single
    ``probability`` band. Tiles are mosaicked spatially within a year
    (outer-join, NaN-filled holes) and concatenated across years on a
    ``time`` dimension. The result is always a :class:`xarray.DataTree`
    with one child per commodity, even when only one commodity is present.

    Example:
        ```python
        index = FDPIndex(release="2025b", gcp_project="my-project")
        await index.build()
        index.load()
        tiles = await index.query(bbox=(9, 5, 11, 6), years=2024,
                                  commodities=["coffee"])

        async with FDPReader(gcp_project="my-project") as reader:
            tree = await reader.open(tiles)

        ds = tree["/coffee"].ds  # probability(time, y, x)
        ```
    """

    def __init__(self, gcp_project: str | None = None):
        self.gcp_project = gcp_project
        self._stores: dict[str, object] = {}
        self._registry: ObjectStoreRegistry | None = None

    async def __aenter__(self) -> FDPReader:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._stores.clear()
        self._registry = None

    def _get_store(self, protocol: PathProtocol, bucket: str):
        # FDP is GCS-only; protocol arg is kept for symmetry with the AEF
        # reader and to surface a clear error if a non-gs path leaks in.
        if protocol != "gs":
            raise ValueError(
                f"FDPReader only supports gs:// paths; got protocol {protocol!r}"
            )
        store_key = f"{protocol}://{bucket}"
        if store_key not in self._stores:
            self._stores[store_key] = make_gcs_store(bucket, self.gcp_project)
        return self._stores[store_key]

    def _get_registry(self, protocol: PathProtocol, bucket: str):
        if self._registry is None:
            self._registry = ObjectStoreRegistry()
        bucket_url = f"{protocol}://{bucket}/"
        store = self._get_store(protocol, bucket)
        self._registry.register(bucket_url, store)
        return self._registry

    async def open(
        self,
        tiles: list[FDPTileInfo],
        ifd: int = 0,
        chunks: int | dict | Literal["auto"] | None = "auto",
    ) -> DataTree:
        """Open FDP tiles into a DataTree grouped by commodity.

        Args:
            tiles: Tiles from :meth:`FDPIndex.query`.
            ifd: Image File Directory index (0 for full resolution).
            chunks: Forwarded to :func:`xarray.open_zarr`. ``"auto"`` (the
                default) lets dask pick chunk sizes; pass ``None`` to load
                eagerly.

        Returns:
            DataTree with one child per commodity. Each child Dataset has
            ``probability(time, y, x)`` and ``_FillValue=nodata=NaN``.
        """
        if not tiles:
            raise ValueError("No tiles provided")

        by_commodity: dict[str, list[FDPTileInfo]] = defaultdict(list)
        for t in tiles:
            by_commodity[t.commodity].append(t)

        logger.info(
            f"Loading {len(tiles)} tiles across {len(by_commodity)} commodities: "
            f"{list(by_commodity.keys())}"
        )

        children: dict[str, xr.Dataset] = {}
        for commodity, ctiles in by_commodity.items():
            logger.info(f"Processing commodity {commodity}: {len(ctiles)} tiles")
            ds = await self._combine_single_commodity(ctiles, ifd, chunks)
            ds = assign_crs(ds, _FDP_CRS)
            ds.attrs["commodity"] = commodity
            ds.attrs["num_tiles"] = len(ctiles)
            children[commodity] = ds

        tree_dict = {f"/{c}": ds for c, ds in children.items()}
        tree = DataTree.from_dict(tree_dict)
        tree.attrs["total_tiles"] = len(tiles)
        tree.attrs["commodities"] = list(children.keys())
        return tree

    async def _combine_single_commodity(
        self,
        tiles: list[FDPTileInfo],
        ifd: int,
        chunks: int | dict | Literal["auto"] | None,
    ) -> xr.Dataset:
        parser = VirtualTIFF(ifd=ifd)
        per_tile = await asyncio.gather(
            *[self._open_tile(tile, parser, chunks) for tile in tiles]
        )
        return self._combine_opened_datasets(per_tile)

    async def _open_tile(
        self,
        tile: FDPTileInfo,
        parser: VirtualTIFF,
        chunks: int | dict | Literal["auto"] | None,
    ) -> xr.Dataset:
        protocol, bucket, key = parse_cloud_path(tile.path)
        file_url = f"{protocol}://{bucket}/{key}"
        registry = self._get_registry(protocol, bucket)

        manifest_store = await asyncio.to_thread(
            parser, url=file_url, registry=registry
        )
        ds: xr.Dataset = await asyncio.to_thread(
            xr.open_zarr,
            manifest_store,
            zarr_format=3,
            consolidated=False,
            chunks=chunks,
            mask_and_scale=False,
        )

        # FDP tiles are single-band; rename to a stable name. Catch the
        # multi-var case explicitly rather than silently picking [0].
        if len(ds.data_vars) != 1:
            raise ValueError(
                f"FDP tile {tile.id} expected exactly one data variable, "
                f"got {list(ds.data_vars)}"
            )
        src_var = next(iter(ds.data_vars))
        ds = ds.rename({src_var: "probability"})

        geobox = get_geobox_from_dataset(ds, _FDP_CRS)
        # Force x/y naming — xr_coords defaults to longitude/latitude for
        # geographic CRSs like EPSG:4326, but virtual-tiff produces a dataset
        # whose spatial dims are already x/y.
        coords = xr_coords(geobox, dims=("y", "x"))
        ds = ds.assign_coords(x=coords["x"].values, y=coords["y"].values)
        ds = ds.expand_dims(time=[tile.as_datetime])

        ds["probability"] = set_aef_nodata(ds["probability"], nodata=np.nan)
        ds.attrs["_source_url"] = file_url
        ds.attrs["_tile_id"] = tile.id
        ds.attrs["commodity"] = tile.commodity

        return ds

    @staticmethod
    def _combine_opened_datasets(per_tile_datasets: list[xr.Dataset]) -> xr.Dataset:
        """Mosaic + temporal concat for already-opened FDP tiles.

        Pure xarray, no IO — exposed for unit tests that build synthetic
        datasets to exercise the combine logic without virtual-tiff/obstore.
        """
        if not per_tile_datasets:
            raise ValueError("No datasets to combine")

        datasets_by_time: dict[dt.datetime, list[xr.Dataset]] = defaultdict(list)
        for ds in per_tile_datasets:
            time_val = ds.coords["time"].values[0]
            time_key = dt.datetime.fromtimestamp(
                time_val.astype("datetime64[s]").astype("int"), tz=dt.UTC
            )
            datasets_by_time[time_key].append(ds)

        time_slices: list[xr.Dataset] = []
        for time_val in sorted(datasets_by_time.keys()):
            time_dsets = datasets_by_time[time_val]
            if len(time_dsets) == 1:
                spatial = time_dsets[0]
            else:
                # combine_by_coords is typed as Dataset | DataArray; we only
                # ever pass Datasets in, so the result is a Dataset.
                spatial = xr.combine_by_coords(
                    time_dsets,
                    coords="minimal",
                    compat="override",
                    combine_attrs="drop_conflicts",
                    join="outer",
                )
                assert isinstance(spatial, xr.Dataset)
            time_slices.append(spatial)

        if len(time_slices) == 1:
            combined = time_slices[0]
        else:
            combined = xr.concat(
                time_slices,
                dim="time",
                coords="minimal",
                compat="override",
                combine_attrs="drop_conflicts",
            )

        return set_aef_nodata(combined, nodata=np.nan)
