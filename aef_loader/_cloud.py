"""Shared cloud-storage and CRS helpers used by both AEF and FDP loaders.

This module centralises the small primitives that would otherwise be duplicated
between ``index.py``/``reader.py`` and the FDP submodule: cloud path parsing,
obstore Store construction, and the GeoTIFF affine extraction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import xarray as xr
from affine import Affine
from obstore.store import GCSStore, S3Store

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox

PathProtocol = Literal["gs", "s3"]


def default_cache_dir() -> Path:
    """User-level cache directory for downloaded/built indexes.

    Honours ``XDG_CACHE_HOME`` if set, otherwise ``~/.cache/aef-loader``.
    """
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "aef-loader"


def parse_gcs_path(path: str) -> tuple[str, str]:
    """Parse ``gs://bucket/key`` into ``(bucket, key)``."""
    if path.startswith("gs://"):
        path = path[5:]
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def parse_s3_path(path: str) -> tuple[str, str]:
    """Parse ``s3://bucket/key`` into ``(bucket, key)``."""
    if path.startswith("s3://"):
        path = path[5:]
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def detect_protocol(path: str) -> PathProtocol:
    """Return the cloud protocol (``"gs"`` or ``"s3"``) for a URL."""
    if path.startswith("gs://"):
        return "gs"
    if path.startswith("s3://"):
        return "s3"
    raise ValueError(f"Unknown protocol for path: {path}. Expected gs:// or s3://")


def parse_cloud_path(path: str) -> tuple[PathProtocol, str, str]:
    """Parse a cloud URL into ``(protocol, bucket, key)``."""
    protocol = detect_protocol(path)
    if protocol == "gs":
        bucket, key = parse_gcs_path(path)
    else:
        bucket, key = parse_s3_path(path)
    return protocol, bucket, key


def make_gcs_store(bucket: str, gcp_project: str | None) -> GCSStore:
    """Build an obstore ``GCSStore`` for a requester-pays bucket.

    Args:
        bucket: GCS bucket name (no ``gs://`` prefix).
        gcp_project: Project to bill for egress. Required for requester-pays.
    """
    if not gcp_project:
        raise ValueError(
            "gcp_project is required for accessing GCS requester-pays buckets"
        )
    return GCSStore(
        bucket=bucket,
        client_options={"default_headers": {"x-goog-user-project": gcp_project}},
    )


def make_s3_store(bucket: str, region: str, skip_signature: bool = True) -> S3Store:
    """Build an obstore ``S3Store``. Defaults to anonymous (Source Coop)."""
    return S3Store(bucket=bucket, region=region, skip_signature=skip_signature)


def _affine_from_pixel_scale_and_tiepoint(
    pixel_scale: tuple[float, float, float],
    tiepoint: tuple[float, float, float, float, float, float],
    flip_y: bool = False,
) -> Affine:
    """Build an affine from ``ModelPixelScaleTag`` + ``ModelTiepointTag``.

    Args:
        pixel_scale: ``(sx, sy, sz)`` from the GeoTIFF tag (always positive).
        tiepoint: ``(i, j, k, x, y, z)`` from the GeoTIFF tag — pixel ``(i,j)``
            maps to world ``(x, y)``.
        flip_y: When ``True``, negate the y-step (standard north-up images
            where pixel (0,0) is the NW corner). When ``False`` (the default,
            preserving AEF's existing behaviour), use the raw positive y-step
            from the tag — appropriate for AEF's bottom-up storage.
    """
    sx, sy, _ = pixel_scale
    x, y = tiepoint[3], tiepoint[4]
    return Affine(sx, 0, x, 0, -sy if flip_y else sy, y)


def _affine_from_model_transform(model_transform: tuple[float, ...]) -> Affine:
    """Build an affine from a 4x4 row-major ``ModelTransformationTag``."""
    return Affine(
        model_transform[0],
        model_transform[1],
        model_transform[3],
        model_transform[4],
        model_transform[5],
        model_transform[7],
    )


def get_geobox_from_dataset(ds: xr.Dataset, crs: str, flip_y: bool = False) -> GeoBox:
    """Extract a GeoBox from a dataset's GeoTIFF tag attrs.

    Looks at the data variables for ``model_transformation`` (preferred) or
    ``model_pixel_scale`` + ``model_tiepoint``. ``flip_y`` is only consulted
    on the pixel-scale path; ``model_transformation`` already encodes the
    correct sign.
    """
    from odc.geo.geobox import GeoBox

    height = ds.sizes["y"]
    width = ds.sizes["x"]

    for var in ds.data_vars:
        attrs = ds[var].attrs
        if "model_pixel_scale" in attrs or "model_transformation" in attrs:
            break
    else:
        raise ValueError(
            "Dataset missing model_pixel_scale or model_transformation attribute"
        )

    if "model_transformation" in attrs:
        affine = _affine_from_model_transform(attrs["model_transformation"])
    else:
        affine = _affine_from_pixel_scale_and_tiepoint(
            attrs["model_pixel_scale"],
            attrs.get("model_tiepoint", [0, 0, 0, 0, 0, 0]),
            flip_y=flip_y,
        )

    return GeoBox(shape=(height, width), affine=affine, crs=crs)
