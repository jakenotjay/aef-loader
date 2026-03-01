"""
Utility functions for AEF data processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from odc.geo.xr import xr_reproject
from xarray import DataTree

from aef_loader.constants import (
    AEF_DEQUANT_DIVISOR,
    AEF_NODATA_VALUE,
)

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox


def dequantize_aef(
    data: np.ndarray | xr.DataArray | xr.Dataset,
    divisor: float = AEF_DEQUANT_DIVISOR,
    nodata_value: int = AEF_NODATA_VALUE,
) -> np.ndarray | xr.DataArray | xr.Dataset:
    """
    Dequantize AEF embeddings from int8 to float32.

    AEF embeddings are stored as quantized int8 values [-127, 127].
    This function converts them back to float32 [-1, 1] for use in ML pipelines.

    The formula is: ((value / 127.5) ** 2) * sign(value)

    NoData values (-128) are automatically converted to NaN.

    Args:
        data: Quantized embedding data (int8)
        divisor: Dequantization divisor (default: 127.5)
        nodata_value: Value to treat as nodata (default: -128)

    Returns:
        Dequantized float32 data in range [-1, 1], with NaN for nodata

    Example:
        ```python
        import numpy as np
        quantized = np.array([127, -127, 0, -128], dtype=np.int8)
        dequantized = dequantize_aef(quantized)
        print(dequantized)  # [~1.0, ~-1.0, 0.0, nan]
        ```
    """
    if isinstance(data, xr.Dataset):
        return data.map(lambda x: dequantize_aef(x, divisor, nodata_value))

    # Create nodata mask before conversion
    nodata_mask = data == nodata_value

    # Apply the correct formula: (v/127.5)² × sign(v)
    normalized = data.astype(np.float32) / divisor
    dequantized = (normalized**2) * np.sign(data)

    # Apply nodata mask (convert -128 to NaN)
    if isinstance(data, xr.DataArray):
        dequantized = xr.where(nodata_mask, np.nan, dequantized)
        result = xr.DataArray(
            dequantized,
            dims=data.dims,
            coords=data.coords,
            attrs=data.attrs.copy(),
        )
        result.attrs["units"] = "embedding"
        result.attrs["dequantized"] = True
        result.attrs["_FillValue"] = np.nan
        return result
    else:
        dequantized = np.where(nodata_mask, np.nan, dequantized)

    return dequantized


def quantize_aef(
    data: np.ndarray | xr.DataArray,
    divisor: float = AEF_DEQUANT_DIVISOR,
) -> np.ndarray | xr.DataArray:
    """
    Quantize float32 embeddings to int8 for storage.

    This is the inverse of dequantize_aef().
    Dequantization: ((v / 127.5) ** 2) * sign(v)
    Quantization (inverse): sign(v) * sqrt(|v|) * 127.5

    Args:
        data: Float32 embedding data in range [-1, 1]
        divisor: Quantization divisor (default: 127.5)

    Returns:
        Quantized int8 data in range [-127, 127]
    """
    sign = np.sign(data)
    magnitude = np.sqrt(np.abs(data))
    quantized = np.round(sign * magnitude * divisor)

    # Clamp to valid range [-127, 127] BEFORE casting to int8
    # This prevents overflow (128 -> -128 in int8)
    quantized = np.clip(quantized, -127, 127).astype(np.int8)

    if isinstance(data, xr.DataArray):
        result = xr.DataArray(
            quantized,
            dims=data.dims,
            coords=data.coords,
            attrs=data.attrs.copy(),
        )
        result.attrs["quantized"] = True
        return result

    return quantized


def mask_nodata(
    data: np.ndarray | xr.DataArray,
    nodata_value: int = AEF_NODATA_VALUE,
) -> np.ndarray | xr.DataArray:
    """
    Mask NoData values (-128) in AEF embeddings.

    NoData pixels have -128 in all channels. This function replaces
    NoData values with NaN for proper handling in analysis.

    Args:
        data: AEF embedding data (int8)
        nodata_value: Value to mask (default: -128)

    Returns:
        Data with NoData values replaced by NaN
    """
    if isinstance(data, xr.DataArray):
        return data.where(data != nodata_value)
    return np.where(data == nodata_value, np.nan, data.astype(np.float32))


def split_bands(ds: xr.Dataset, var: str = "embeddings") -> xr.Dataset:
    """
    Split a single multi-band DataArray into separate named variables (A00–A63).

    This is the inverse of the compact band representation used by
    VirtualTiffReader.open_tiles_by_zone(). Use this when downstream code
    expects individual A00–A63 data variables.

    Args:
        ds: Dataset containing a variable with a 'band' dimension
        var: Name of the variable to split (default: "embeddings")

    Returns:
        Dataset with one variable per band (A00, A01, ..., A63)
    """
    da = ds[var]
    split = da.to_dataset(dim="band")
    split.attrs = ds.attrs.copy()
    return split


def reproject_datatree(
    tree: DataTree,
    target_geobox: GeoBox,
    resampling: str = "nearest",
) -> xr.Dataset:
    """
    Reproject all zones in a DataTree to a common target GeoBox.

    This function takes a DataTree with multiple UTM zones and reprojects each
    zone's dataset to a common coordinate system defined by the target GeoBox.
    The reprojected datasets are then combined into a single dataset.

    The reprojection is lazy - it builds a dask computation graph that only
    executes when .compute() is called. Chunks are loaded and reprojected
    on-demand.

    For combining zones, this uses xarray's combine_first which:
    - Uses values from earlier zones where available (non-NaN)
    - Fills NaN regions with values from subsequent zones
    - In true overlapping regions (both have valid data), earlier zones take precedence

    Since overlapping regions contain reprojections of the same underlying data,
    values should be identical regardless of which zone they come from.

    Args:
        tree: DataTree with zone datasets as children (from open_tiles_by_zone)
        target_geobox: Target GeoBox defining the output CRS, resolution, and extent.
                       Can be created with GeoBox.from_bbox() or from an existing dataset.
        resampling: Resampling method - "nearest", "bilinear", "cubic", etc.
                    Default is "nearest" which preserves original int8 values.

    Returns:
        Combined xr.Dataset with all zones reprojected to the target GeoBox.
        Data variables remain as dask arrays until .compute() is called.

    Example:
        ```python
        from odc.geo.geobox import GeoBox

        # Create target geobox (e.g., 100m resolution in EPSG:4326)
        target = GeoBox.from_bbox(
            bbox=(-122.5, 37.5, -121.5, 38.5),
            crs="EPSG:4326",
            resolution=0.001,  # ~100m at this latitude
        )

        # Reproject all zones to target
        combined = reproject_datatree(tree, target)
        result = combined.compute()  # triggers actual reprojection
        ```
    """
    reprojected_datasets = []

    for zone_name in tree.children:
        zone_ds = tree[zone_name].ds

        # Skip empty datasets
        if zone_ds is None or len(zone_ds.data_vars) == 0:
            continue

        # Reproject to target geobox (lazy operation with dask)
        reprojected = xr_reproject(zone_ds, target_geobox, resampling=resampling)

        # Add source zone as attribute
        reprojected.attrs["source_zone"] = zone_name
        reprojected_datasets.append(reprojected)

    if len(reprojected_datasets) == 0:
        raise ValueError("No datasets to reproject")

    if len(reprojected_datasets) == 1:
        return reprojected_datasets[0]

    # Combine datasets using combine_first. Each reprojected dataset has NaN
    # where its source zone had no coverage. combine_first fills NaN values
    # from the first dataset with valid values from subsequent datasets.
    # In true overlapping regions where both have valid data, the first
    # dataset's values are used - but since they're reprojections of the
    # same underlying data, values should be identical.
    combined = reprojected_datasets[0]
    for ds in reprojected_datasets[1:]:
        combined = combined.combine_first(ds)

    combined.attrs["source_zones"] = [
        tree[z].ds.attrs.get("utm_zone", z) for z in tree.children
    ]
    combined.attrs["target_crs"] = str(target_geobox.crs)

    return combined
