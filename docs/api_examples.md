# API Reference

## AEFIndex

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

## VirtualTiffReader

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

## DataSource

Enum for selecting the data backend.

```python
from aef_loader import DataSource

DataSource.SOURCE_COOP  # AWS S3 (free, no auth)
DataSource.GCS          # Google Cloud Storage (requester pays)
```

## Utility Functions

### `mask_nodata`

Mask NoData values (-128) before processing.

```python
from aef_loader import mask_nodata

masked = mask_nodata(data)
```

### `dequantize_aef`

Dequantize int8 embeddings to float32.

Formula: `((value / 127.5) ** 2) * sign(value)`

```python
from aef_loader import dequantize_aef

float_data = dequantize_aef(data)
```

### `quantize_aef`

Quantize float32 embeddings back to int8 for storage. Useful after dequantizing, reprojecting with bilinear interpolation, and re-quantizing.

```python
from aef_loader import quantize_aef

int8_data = quantize_aef(float_data)
```

### `reproject_datatree`

Reproject all zones in a DataTree to a common CRS.

```python
from aef_loader.utils import reproject_datatree
from odc.geo.geobox import GeoBox

target = GeoBox.from_bbox(
    bbox=(-122.5, 37.5, -122.0, 38.0),
    crs="EPSG:4326",
    resolution=0.0001,  # ~10m
)
combined = reproject_datatree(tree, target)
```

### `split_bands`

Split a multi-band dataset into individual band variables.

```python
from aef_loader import split_bands

split = split_bands(dataset)
```
