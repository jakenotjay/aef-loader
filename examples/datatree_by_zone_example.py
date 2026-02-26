"""
Load AEF tiles as a DataTree organized by UTM zone.

Demonstrates:
1. Querying tiles for a geographic region (e.g., a country or complex geometry)
2. Loading tiles as a DataTree with zones as groups
3. Clipping to a complex geometry (only loading necessary chunks)
4. Iterating over zones to access data in native CRS
5. Applying dequantization lazily

This is the recommended approach when loading tiles that span multiple UTM zones,
since tiles in different zones have different CRS and cannot be combined without
reprojection. The consumer can then reproject each zone to a common grid using
odc.geobox or similar.
"""

from __future__ import annotations

import asyncio

import geopandas as gpd
import numpy as np
import xarray as xr
from aef_loader.constants import (
    AEF_DEQUANT_DIVISOR,
    AEF_NODATA_VALUE,
    DataSource,
)
from aef_loader.index import AEFIndex
from aef_loader.reader import VirtualTiffReader
from aef_loader.types import BoundingBox
from aef_loader.utils import clip_to_geometry, reproject_datatree
from odc.geo.geobox import GeoBox
from shapely.geometry import box, shape

# Complex geometry for clipping (San Francisco Bay Area)
CLIP_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [
            [-122.0652539996401, 37.913423635634956],
            [-122.30871529431084, 37.832744251305755],
            [-122.20640212175029, 37.672486912577554],
            [-122.34045133678325, 37.67197433957509],
            [-122.33445305888009, 37.53716078072131],
            [-122.10991273765094, 37.512907437699866],
            [-121.93576190802423, 37.64696471573727],
            [-122.04718740362065, 37.65405821356357],
            [-121.90871149979236, 37.84540096427328],
            [-122.0652539996401, 37.913423635634956],
        ]
    ],
}


def dequantize_lazy(data: xr.DataArray) -> xr.DataArray:
    """Lazily dequantize int8 -> float32. Formula: ((v/127.5)^2 * sign(v))"""
    nodata_mask = data == AEF_NODATA_VALUE
    normalized = data.astype(np.float32) / AEF_DEQUANT_DIVISOR
    dequantized = (normalized**2) * np.sign(data)
    return xr.where(nodata_mask, np.nan, dequantized)


async def main():
    print("Load AEF tiles as DataTree by UTM zone")
    print("=" * 50)

    # Use the geometry bounds as bbox for tile query
    geometry = shape(CLIP_GEOMETRY)
    bbox: BoundingBox = geometry.bounds
    print(f"Geometry bounds: {bbox}")

    # Initialize index and load tiles
    index = AEFIndex(source=DataSource.SOURCE_COOP)
    await index.download()
    index.load()
    tiles = await index.query(bbox=bbox, years=(2020, 2022))

    if not tiles:
        print("No tiles found")
        return

    print(f"Found {len(tiles)} tiles")

    # Export queried tiles and clip geometry to GeoJSON for QGIS visualization
    tiles_gdf = gpd.GeoDataFrame(
        [
            {
                "id": t.id,
                "year": t.year,
                "utm_zone": t.utm_zone,
                "crs": f"EPSG:{t.crs_epsg}",
                "path": t.path,
            }
            for t in tiles
        ],
        geometry=[box(*t.bbox) for t in tiles],
        crs="EPSG:4326",
    )
    tiles_gdf.to_file("queried_tiles.geojson", driver="GeoJSON")
    print("Exported queried tiles to queried_tiles.geojson")

    # Export clip geometry
    clip_gdf = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
    clip_gdf.to_file("clip_geometry.geojson", driver="GeoJSON")
    print("Exported clip geometry to clip_geometry.geojson")

    # Show tile distribution by zone
    zones = {}
    for tile in tiles:
        zone = tile.utm_zone or "unknown"
        zones[zone] = zones.get(zone, 0) + 1
    print(f"Tiles by zone: {zones}")

    # Load as DataTree organized by UTM zone
    print("\nLoading tiles into DataTree by zone...")
    async with VirtualTiffReader() as reader:
        tree = await reader.open_tiles_by_zone(tiles)

    # Inspect the DataTree structure
    print("\nDataTree structure:")
    print(tree)
    print(f"  Root attributes: {dict(tree.attrs)}")
    print(f"  Zones: {tree.attrs.get('zones', [])}")

    # Iterate over each zone
    for zone_name in tree.children:
        zone_ds = tree[zone_name].ds
        print(zone_ds)
        print(f"\n  Zone {zone_name}:")
        print(f"    Dimensions: {dict(zone_ds.sizes)}")
        print(f"    Num tiles: {zone_ds.attrs.get('num_tiles')}")

        # Show coordinate ranges (UTM meters)
        if "x" in zone_ds.coords and "y" in zone_ds.coords:
            x = zone_ds.coords["x"].values
            y = zone_ds.coords["y"].values
            print(f"    X range: {x.min():.0f} to {x.max():.0f} m")
            print(f"    Y range: {y.min():.0f} to {y.max():.0f} m")

    # Reproject all zones to a common target GeoBox
    print("\n" + "=" * 50)
    print("Reprojecting to target GeoBox...")

    # Create target GeoBox from the geometry bounds
    # Using EPSG:4326 (WGS84) at ~10m resolution for this demo
    # In practice, you might use a UTM zone or other projection
    target_geobox = GeoBox.from_bbox(
        bbox=geometry.bounds,
        crs="EPSG:4326",
        resolution=0.0001,  # ~10m at this latitude
    )
    print(f"Target GeoBox: {target_geobox.shape} pixels at {target_geobox.resolution}")

    # Reproject all zones to target (lazy operation with dask)
    reprojected = reproject_datatree(tree, target_geobox, resampling="nearest")

    print(f"Reprojected dimensions: {dict(reprojected.sizes)}")
    print(f"Reprojected CRS: {reprojected.odc.crs}")

    # Clip to complex geometry
    print("\n" + "=" * 50)
    print("Clipping to complex geometry...")

    print(f"Before clipping: {dict(reprojected.sizes)}")

    # Clip to geometry - only loads chunks that intersect
    clipped = clip_to_geometry(reprojected, CLIP_GEOMETRY)

    print(f"After clipping: {dict(clipped.sizes)}")

    # Select single time step for demo
    if "time" in clipped.dims:
        clipped = clipped.isel(time=0, drop=True)

    # Load the data (triggers actual reprojection and reads from remote COGs)
    print("\nLoading clipped and reprojected data...")
    result = clipped.compute()

    # Count valid pixels
    sample_var = list(result.data_vars)[0]
    valid_mask = ~np.isnan(result[sample_var].values)
    total_pixels = valid_mask.size
    valid_pixels = valid_mask.sum()
    print(
        f"Valid pixels: {valid_pixels:,} / {total_pixels:,} ({100 * valid_pixels / total_pixels:.1f}%)"
    )

    # Show stats for a few variables
    print("\nStats for first 5 bands:")
    for var in list(result.data_vars)[:5]:
        arr = result[var]
        valid = arr.values[~np.isnan(arr.values)]
        if len(valid) > 0:
            print(f"  {var}: min={valid.min()}, max={valid.max()}")

    # Apply dequantization
    print("\nApplying dequantization...")
    dequantized = result.map(dequantize_lazy)

    # Save to zarr
    dequantized.to_zarr("dequantized_reprojected.zarr", mode="w")
    print("Saved to dequantized_reprojected.zarr")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
