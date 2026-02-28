# aef-loader

Virtualizarr access for AEF embeddings as an analysis ready data cube. 5x quicker than rasterio access, many x quicker than earth engine.

## TODOs before v0.1.0
- [x] Cleanup from monorepo
- [x] Cleanup unused code, quality pass for AI gen nonsense
- [x] Ensure tests are relevant
- [x] CI/CD (mostly follow https://github.com/carderne/postmodern-python)
- [x] precommit for CI/CD
- [ ] Fix examples up, add notebooks (do people still use these?)
- [ ] Proper documentation, friendly readme
- [x] read the docs
- [ ] benchmarks: rasterio/rioxarray, XEE with HVE (20 vs 500 connections)
- [ ] reprojection best practices i.e. when do you need to dequantise before projection?
- [ ] blog post (why? virtual what now? learnings about odc-geo reprojection and GIL contention, reprojection nonsense, dtype nonsense, dask nonsense, requester pays nonsense, thanks to vtif and obstore folks, why geobox, what i like about it)
- [ ] also include the origin of why, high volume endpoint and Xee is very slow for pulling covariates because it A) generates the coordinates via API request rather than linear algebra B) auto chunks such such that images are pulled 1 band, 256 x 256 chunks + costs money
- [ ] cleanup dependencies (we don't need dask here)

## Overview

Alpha Earth Foundations embeddings is a dataset produced by Google Deepmind, providing a yearly 64-channel embeddings derived from numerous satellite image sources with numerous downstream applications. The embeddings are stored as multi-band Cloud-Optimised GeoTIFFs (COGs).

aef-loader supports two dataset hosts, both having tradeoffs:
1. [Google Cloud Storage](https://developers.google.com/earth-engine/guides/aef_on_gcs_readme) - maintained by the [Earth Engine](https://earthengine.google.com/) team, more up to date but requiring authentication and "[requester pays](https://docs.cloud.google.com/storage/docs/requester-pays)", meaning users must pay egress and other charges.
2. [Source Cooperative](https://source.coop/tge-labs/aef) - Hosted on AWS S3 and free to access, but generally less up to date (currently missing 2017 and 2025)

## Attribution and Dataset License
This dataset is licensed under CC-BY 4.0 and requires the following attribution text: "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."

AEF provides 64-channel satellite embeddings derived from satellite imagery that can be used for various geospatial ML tasks. The embeddings are stored as multi-band Cloud-Optimized GeoTIFFs (COGs) on Google Cloud Storage and Source Cooperative (AWS S3).

This package provides:

- **Lazy loading** - Read COGs as virtual zarr stores without copying data
- **Spatial/temporal queries** - Filter tiles by bounding box and year range using the GeoParquet index
- **UTM zone organization** - Data organized by native UTM projection via DataTree
- **Reprojection utilities** - Combine data across UTM zones to a common CRS

## Installation

```bash
uv sync
```

## Data Sources

The package supports two data sources:

### Source Cooperative (Recommended)
- **No authentication required** - Public bucket on AWS S3
- **URL**: `s3://us-west-2.opendata.source.coop/tge-labs/aef/`
- **Note**: 2017 data not available

### Google Cloud Storage
- **Requires GCP credentials** - Requester-pays bucket
- **URL**: `gs://alphaearth_foundations/`

## Quick Start

```python
import asyncio
from aef_loader import AEFIndex, VirtualTiffReader, DataSource
from aef_loader.utils import reproject_datatree
from odc.geo.geobox import GeoBox

async def main():
    # Initialize index (Source Cooperative - no auth needed)
    index = AEFIndex(source=DataSource.SOURCE_COOP)
    await index.download()
    index.load()

    # Query for tiles
    tiles = await index.query(
        bbox=(-122.5, 37.5, -122.0, 38.0),
        years=(2020, 2023),
    )

    # Load tiles organized by UTM zone
    async with VirtualTiffReader() as reader:
        tree = await reader.open_tiles_by_zone(tiles)

    # Each zone is a separate Dataset with its native CRS
    for zone in tree.children:
        ds = tree[zone].ds
        print(f"{zone}: {ds.odc.crs}, {dict(ds.sizes)}")

    # Optionally reproject all zones to a common CRS
    target = GeoBox.from_bbox(
        bbox=(-122.5, 37.5, -122.0, 38.0),
        crs="EPSG:4326",
        resolution=0.0001,  # ~10m
    )
    combined = reproject_datatree(tree, target)

asyncio.run(main())
```

## API Reference

### AEFIndex

Manages the GeoParquet index for spatial/temporal queries.

```python
from aef_loader import AEFIndex, DataSource

# Source Cooperative (public, no auth)
index = AEFIndex(source=DataSource.SOURCE_COOP)

# Or GCS (requires project for requester-pays)
index = AEFIndex(source=DataSource.GCS, gcp_project="my-project")

# Download the index (cached after first download)
await index.download()

# Load into memory
gdf = index.load()

# Query for tiles
tiles = await index.query(
    bbox=(-122.5, 37.5, -122.0, 38.0),  # WGS84 coordinates
    years=(2020, 2023),                   # Year range
    limit=10,                             # Max tiles
)
```

### VirtualTiffReader

Opens COGs as virtual zarr stores organized by UTM zone.

```python
from aef_loader import VirtualTiffReader

async with VirtualTiffReader() as reader:
    # Load tiles organized by UTM zone
    tree = await reader.open_tiles_by_zone(tiles)

    # Returns a DataTree with zones as children:
    # ├── 10N/  → Dataset with A00-A63 variables in EPSG:32610
    # ├── 10S/  → Dataset with A00-A63 variables in EPSG:32710
    # ├── 11N/  → Dataset with A00-A63 variables in EPSG:32611
    # ...
```

### Utility Functions

```python
from aef_loader import (
    dequantize_aef,
    quantize_aef,
    mask_nodata,
    reproject_datatree,
)

# Mask NoData values (-128) before processing
masked = mask_nodata(data)

# Dequantize int8 embeddings to float32
# Formula: ((value / 127.5) ** 2) * sign(value)
dequantized = dequantize_aef(data)

# Quantize float32 embeddings back to int8 for storage
# e.g. after dequantize -> reproject(bilinear) -> quantize
requantized = quantize_aef(dequantized)

# Reproject all zones in DataTree to common CRS
from odc.geo.geobox import GeoBox
target = GeoBox.from_bbox(bbox=bbox, crs="EPSG:4326", resolution=0.0001)
combined = reproject_datatree(tree, target)
```

## Examples

See `examples/` for complete working examples:

```bash
# Source Cooperative (no auth required)
uv run examples/source_coop_example.py

# Full workflow with reprojection
uv run examples/source_coop_example.py --full-workflow

# GCS (requires project)
uv run examples/aef_embeddings_example.py --project YOUR_PROJECT_ID
```

## Data Format

AEF embeddings are stored as:
- **Format**: Cloud-Optimized GeoTIFF (COG)
- **Channels**: 64 bands per tile (A00-A63)
- **Data type**: int8 (signed, quantized)
- **NoData**: -128
- **Dequantization**: `((value / 127.5) ** 2) * sign(value)`
- **CRS**: UTM zone appropriate to tile location

To use in ML pipelines, dequantize to float32:
```python
# First mask NoData values, then dequantize
masked = mask_nodata(int8_embeddings)
float_embeddings = dequantize_aef(masked)
```

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
pytest

# Run tests with coverage
pytest --cov=aef_loader
```

## Dependencies

- **virtual-tiff** - Creates virtual zarr stores from TIFFs
- **virtualizarr** - Combines multiple COGs into data cubes
- **obstore** - Async GCS/S3 access
- **odc-geo** - Geospatial coordinate handling and reprojection
- **xarray** - N-dimensional labeled arrays
- **dask** - Parallel/lazy computation
- **geopandas** - Spatial data handling

## Links

- [AEF Documentation](https://developers.google.com/earth-engine/guides/aef_on_gcs_readme)
- [Source Cooperative](https://source.coop/tge-labs/aef)
- [virtual-tiff Documentation](https://virtual-tiff.readthedocs.io/)
