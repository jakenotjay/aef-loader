"""
COG reader using virtual-tiff library.

Provides efficient COG reading by virtualizing TIFFs as Zarr stores.
Supports both GCS (Google Cloud Storage) and S3 (Source Cooperative) backends.

The primary access pattern is loading tiles organized by UTM zone using
`open_tiles_by_zone()`. For combining data across zones, use
`reproject_datatree()` from the utils module.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import xarray as xr
from odc.geo.xr import assign_crs, xr_coords
from virtual_tiff import VirtualTIFF
from virtualizarr.registry import ObjectStoreRegistry
from xarray import DataTree

from aef_loader._cloud import (
    PathProtocol,
    get_geobox_from_dataset,
    make_gcs_store,
    make_s3_store,
    parse_cloud_path,
)
from aef_loader.constants import SOURCE_COOP_REGION
from aef_loader.utils import set_aef_nodata

if TYPE_CHECKING:
    from aef_loader.types import AEFTileInfo

logger = logging.getLogger(__name__)


class VirtualTiffReader:
    """
    COG reader using virtual-tiff to create virtual zarr stores.

    This provides efficient COG access by:
    - Creating virtual zarr stores from COGs without data duplication
    - Using async I/O via obstore for cloud access (GCS and S3)
    - Organizing tiles by UTM zone for proper CRS handling
    - Integrating directly with xarray for data loading

    The primary method is `open_tiles_by_zone()` which loads tiles organized
    by their native UTM zone. To combine data across zones, use
    `reproject_datatree()` from the utils module.

    Example:
        ```python
        from aef_loader import AEFIndex, VirtualTiffReader, DataSource
        from aef_loader.utils import reproject_datatree
        from odc.geo.geobox import GeoBox

        # Query tiles
        index = AEFIndex(source=DataSource.SOURCE_COOP)
        await index.download()
        index.load()
        tiles = await index.query(bbox=(-122.5, 37.5, -121.5, 38.5), years=(2020, 2022))

        # Load by UTM zone
        async with VirtualTiffReader() as reader:
            tree = await reader.open_tiles_by_zone(tiles)

        # Reproject to common CRS if needed
        target = GeoBox.from_bbox(bbox=(-122.5, 37.5, -121.5, 38.5), crs="EPSG:4326", resolution=0.0001)
        combined = reproject_datatree(tree, target)
        result = combined.compute()
        ```
    """

    def __init__(
        self,
        gcp_project: str | None = None,
    ):
        """
        Initialize the virtual TIFF reader.

        Args:
            gcp_project: GCP project ID for requester-pays buckets (GCS only)
        """
        self.gcp_project = gcp_project
        self._stores: dict[str, object] = {}  # Cache stores by (protocol, bucket)
        self._registry = None

    async def __aenter__(self) -> VirtualTiffReader:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._stores.clear()
        self._registry = None

    def _get_store(self, protocol: PathProtocol, bucket: str):
        """Get or create an obstore for a bucket based on protocol."""
        store_key = f"{protocol}://{bucket}"
        if store_key not in self._stores:
            if protocol == "gs":
                self._stores[store_key] = make_gcs_store(bucket, self.gcp_project)
            elif protocol == "s3":
                self._stores[store_key] = make_s3_store(bucket, SOURCE_COOP_REGION)
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")
        return self._stores[store_key]

    def _get_registry(self, protocol: PathProtocol, bucket: str):
        """Get or create an ObjectStoreRegistry for a bucket."""
        if self._registry is None:
            self._registry = ObjectStoreRegistry()

        bucket_url = f"{protocol}://{bucket}/"
        store = self._get_store(protocol, bucket)
        self._registry.register(bucket_url, store)

        return self._registry

    async def open_tiles_by_zone(
        self,
        tiles: list[AEFTileInfo],
        ifd: int = 0,
        chunks: int | dict | Literal["auto"] | None = "auto",
    ) -> DataTree:
        """
        Open tiles and organize them by UTM zone in a DataTree.

        Each UTM zone becomes a group in the DataTree, containing a Dataset
        with a single 'embeddings' variable with a band dimension (A00–A63).
        Both ``nodata`` and ``_FillValue`` attrs are set to ``-128`` on each
        embeddings variable so that downstream tools (odc-geo ``xr_reproject``,
        xarray) correctly identify the AEF nodata sentinel.

        This is the primary method for loading AEF data. It keeps each zone's
        data in its native CRS for accurate spatial operations. To combine
        data across zones, use `reproject_datatree()` from the utils module.

        Args:
            tiles: List of AEFTileInfo objects from AEFIndex.query()
            ifd: Image File Directory index (0 for full resolution)
            chunks: The chunks parameter to pass to open_zarr, defaults to auto,
                useful to pass None to stop dask task explosions

        Returns:
            DataTree with structure:
                ├── 10N/  → Dataset with embeddings(time, band, y, x) in EPSG:32610
                ├── 10S/  → Dataset with embeddings(time, band, y, x) in EPSG:32710
                ├── 11N/  → Dataset with embeddings(time, band, y, x) in EPSG:32611
                ...

        Example:
            ```python
            tiles = await index.query(bbox=bbox, years=(2020, 2022))
            async with VirtualTiffReader() as reader:
                tree = await reader.open_tiles_by_zone(tiles)
            for zone in tree.children:
                ds = tree[zone].ds
                print(f"{zone}: {ds.odc.crs}, {dict(ds.sizes)}")
            ```
        """
        if not tiles:
            raise ValueError("No tiles provided")

        # Group tiles by UTM zone
        tiles_by_zone: dict[str, list[AEFTileInfo]] = defaultdict(list)
        for tile in tiles:
            zone = tile.utm_zone or "unknown"
            tiles_by_zone[zone].append(tile)

        logger.info(
            f"Loading {len(tiles)} tiles across {len(tiles_by_zone)} UTM zones: "
            f"{list(tiles_by_zone.keys())}"
        )

        # Process each zone
        zone_datasets: dict[str, xr.Dataset] = {}
        for zone, zone_tiles in tiles_by_zone.items():
            logger.info(f"Processing zone {zone}: {len(zone_tiles)} tiles")

            # Combine tiles within the zone
            ds = await self._combine_tiles_single_zone(zone_tiles, ifd, chunks=chunks)

            # Add CRS metadata using odc-geo
            crs = f"EPSG:{zone_tiles[0].crs_epsg}"
            ds = assign_crs(ds, crs)

            # Add zone metadata
            ds.attrs["utm_zone"] = zone
            ds.attrs["num_tiles"] = len(zone_tiles)

            zone_datasets[zone] = ds

        # Build DataTree
        tree_dict = {f"/{zone}": ds for zone, ds in zone_datasets.items()}
        tree = DataTree.from_dict(tree_dict)

        # Add root attributes
        tree.attrs["total_tiles"] = len(tiles)
        tree.attrs["zones"] = list(zone_datasets.keys())

        return tree

    async def _combine_tiles_single_zone(
        self,
        tiles: list[AEFTileInfo],
        ifd: int = 0,
        chunks: int | dict | Literal["auto"] | None = "auto",
    ) -> xr.Dataset:
        """
        Combine tiles within a single UTM zone.

        All tiles must be in the same CRS. Combines spatially and temporally,
        keeping bands as a single 'embeddings' variable with a band dimension.
        Sets both ``nodata`` and ``_FillValue`` to ``-128`` on the output via
        ``set_aef_nodata``.
        """
        parser = VirtualTIFF(ifd=ifd)

        async def process_tile(tile: AEFTileInfo) -> xr.Dataset:
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

            # Extract GeoBox from the model_transformation in the TIFF
            # This correctly handles bottom-up images (positive y scale)
            crs = f"EPSG:{tile.crs_epsg}"
            geobox = get_geobox_from_dataset(ds, crs)
            coords = xr_coords(geobox)

            # Assign spatial coordinates from the actual TIFF affine
            ds = ds.assign_coords(x=coords["x"].values, y=coords["y"].values)

            # Expand time as a dimension
            ds = ds.expand_dims(time=[tile.as_datetime])

            ds.attrs["_source_url"] = file_url
            ds.attrs["_tile_id"] = tile.id

            return ds

        datasets = await asyncio.gather(*[process_tile(tile) for tile in tiles])

        # Group by time
        datasets_by_time: dict[dt.datetime, list[xr.Dataset]] = defaultdict(list)
        for ds in datasets:
            time_val = ds.coords["time"].values[0]
            time_key = dt.datetime.fromtimestamp(
                time_val.astype("datetime64[s]").astype("int"), tz=dt.UTC
            )
            datasets_by_time[time_key].append(ds)

        # Combine spatially within each time, then temporally
        time_slices = []
        for time_val in sorted(datasets_by_time.keys()):
            time_datasets = datasets_by_time[time_val]

            if len(time_datasets) == 1:
                spatial_combined = time_datasets[0]
            else:
                spatial_combined = xr.combine_by_coords(
                    time_datasets,
                    coords="minimal",
                    compat="override",
                    combine_attrs="drop_conflicts",
                    join="outer",
                )
            time_slices.append(spatial_combined)

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

        # Keep bands as a single variable with string band coordinates
        if "band" in combined.dims:
            data_var = list(combined.data_vars)[0]
            da = combined[data_var]
            # Assign string band coordinate labels (A00, A01, ..., A63)
            band_names = [f"A{i:02d}" for i in range(da.sizes["band"])]
            da = da.assign_coords(band=band_names)
            da.name = "embeddings"
            da = set_aef_nodata(da)
            combined = da.to_dataset()

        return combined
