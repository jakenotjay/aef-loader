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
from affine import Affine
from obstore.store import GCSStore, S3Store
from odc.geo.geobox import GeoBox
from odc.geo.xr import assign_crs, xr_coords
from virtual_tiff import VirtualTIFF
from virtualizarr.registry import ObjectStoreRegistry
from xarray import DataTree

from aef_loader.constants import SOURCE_COOP_REGION

if TYPE_CHECKING:
    from aef_loader.types import AEFTileInfo

logger = logging.getLogger(__name__)

PathProtocol = Literal["gs", "s3"]


def _parse_gcs_path(path: str) -> tuple[str, str]:
    """Parse gs://bucket/key into (bucket, key)."""
    if path.startswith("gs://"):
        path = path[5:]
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _parse_s3_path(path: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    if path.startswith("s3://"):
        path = path[5:]
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _detect_protocol(path: str) -> PathProtocol:
    """Detect the cloud protocol from a path."""
    if path.startswith("gs://"):
        return "gs"
    elif path.startswith("s3://"):
        return "s3"
    else:
        raise ValueError(f"Unknown protocol for path: {path}. Expected gs:// or s3://")


def _parse_cloud_path(path: str) -> tuple[PathProtocol, str, str]:
    """Parse cloud path into (protocol, bucket, key)."""
    protocol = _detect_protocol(path)
    if protocol == "gs":
        bucket, key = _parse_gcs_path(path)
    else:
        bucket, key = _parse_s3_path(path)
    return protocol, bucket, key


def _get_affine_from_model_pixel_scale_and_tiepoint(
    pixel_scale: tuple[float, float, float],
    tiepoint: tuple[float, float, float, float, float, float],
) -> Affine:
    """Creates an affine transform from the pixel scale and tiepoint.

    Args:
        pixel_scale: the ModelPixelScale tag - 3 values representing scale factor of x, y, and z
        tiepoint: the ModelTiepointTag - 6 values representing each of the tiepoints

    Returns:
        An affine transform calculated from these values.
    """
    sx, sy, _ = pixel_scale
    x, y = tiepoint[3], tiepoint[4]

    # TODO: validate the positive sy is correct, as I believe all the AEF images are bottom up
    return Affine(sx, 0, x, 0, sy, y)


def _get_affine_from_model_transform(model_transform: tuple[float, ...]) -> Affine:
    """Creates an affine transform from the model transform.

    Args:
        model_transform: The ModelTransformTag - 4x4 homogeneous transformation matrix in row-major order

    Returns:
        An affine transform calculated from these values
    """
    return Affine(
        model_transform[0],
        model_transform[1],
        model_transform[3],
        model_transform[4],
        model_transform[5],
        model_transform[7],
    )


def _get_geobox_from_dataset(ds: xr.Dataset, crs: str) -> GeoBox:
    """Extract GeoBox from dataset using the model_transformation or model_pixel_scale attribute if available.

    The model_transformation is a 4x4 matrix from the GeoTIFF that defines the
    affine transformation from pixel coordinates to CRS coordinates. This properly
    handles images stored bottom-up (positive y scale).

    Args:
        ds: Dataset with model_transformation in data variable attrs
        crs: CRS string (e.g., "EPSG:32610")

    Returns:
        GeoBox with correct affine transformation
    """
    height = ds.sizes["y"]
    width = ds.sizes["x"]

    for var in ds.data_vars:
        attrs = ds[var].attrs
        if ("model_pixel_scale" in attrs) or "model_transformation" in attrs:
            break
    else:
        raise ValueError(
            "Dataset missing model_pixel_scale or model_transformation attribute"
        )

    if "model_pixel_scale" in attrs:
        affine = _get_affine_from_model_pixel_scale_and_tiepoint(
            attrs["model_pixel_scale"], attrs.get("model_tiepoint", [0, 0, 0, 0, 0, 0])
        )
    else:
        affine = _get_affine_from_model_transform(attrs["model_transformation"])

    return GeoBox(shape=(height, width), affine=affine, crs=crs)


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

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._stores.clear()
        self._registry = None

    def _get_gcs_store(self, bucket: str):
        """Get or create an obstore GCSStore for a bucket."""
        if not self.gcp_project:
            raise ValueError(
                "gcp_project is required for reading from GCS requester-pays bucket"
            )
        return GCSStore(
            bucket=bucket,
            client_options={
                "default_headers": {"x-goog-user-project": self.gcp_project}
            },
        )

    def _get_s3_store(self, bucket: str):
        """Get or create an obstore S3Store for a bucket (Source Cooperative)."""
        return S3Store(
            bucket=bucket,
            region=SOURCE_COOP_REGION,
            skip_signature=True,  # Source Coop is public, no auth needed
        )

    def _get_store(self, protocol: PathProtocol, bucket: str):
        """Get or create an obstore for a bucket based on protocol."""
        store_key = f"{protocol}://{bucket}"
        if store_key not in self._stores:
            if protocol == "gs":
                self._stores[store_key] = self._get_gcs_store(bucket)
            elif protocol == "s3":
                self._stores[store_key] = self._get_s3_store(bucket)
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
            ds = await self._combine_tiles_single_zone(zone_tiles, ifd)

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
        """
        parser = VirtualTIFF(ifd=ifd)

        async def process_tile(tile: AEFTileInfo) -> xr.Dataset:
            protocol, bucket, key = _parse_cloud_path(tile.path)
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
            )

            # Extract GeoBox from the model_transformation in the TIFF
            # This correctly handles bottom-up images (positive y scale)
            crs = f"EPSG:{tile.crs_epsg}"
            geobox = _get_geobox_from_dataset(ds, crs)
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
            combined = da.to_dataset()

        return combined
