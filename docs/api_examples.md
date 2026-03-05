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

Opens COGs as virtual zarr stores organized by UTM zone,
you can pass chunks as a parameter here to control chunking from the start.

For example, it's almost certain you want all your bands together on a single worker,
so you would pass `chunks={"band": -1}` to ensure the band dimension is not split across chunks.
Otherwise costly rechunks/shuffles are required between workers after zones are merged.

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

Dequantize int8 embeddings to float32. NoData values (-128) become NaN.
Sets both `nodata` and `_FillValue` attrs to `NaN` on the output; all other
existing attrs are preserved.

Formula: `((value / 127.5) ** 2) * sign(value)`

```python
from aef_loader import dequantize_aef

float_data = dequantize_aef(data)
# float_data.attrs["nodata"]     -> NaN
# float_data.attrs["_FillValue"] -> NaN
```

### `quantize_aef`

Quantize float32 embeddings back to int8 for storage. Useful after dequantizing, reprojecting with bilinear interpolation, and re-quantizing.
Sets both `nodata` and `_FillValue` attrs to `-128` on the output; all other
existing attrs are preserved.

```python
from aef_loader import quantize_aef

int8_data = quantize_aef(float_data)
# int8_data.attrs["nodata"]     -> -128
# int8_data.attrs["_FillValue"] -> -128
```

### `set_aef_nodata`

Stamp `nodata` and `_FillValue` attrs on a DataArray or Dataset.
Returns a new object (the input is not modified). Defaults to `-128`
(the AEF int8 sentinel); pass `np.nan` for dequantized float data.

```python
from aef_loader import set_aef_nodata

# Raw / quantized embeddings
da = set_aef_nodata(da)              # nodata=-128, _FillValue=-128

# Dequantized float data
da = set_aef_nodata(da, nodata=np.nan)  # nodata=NaN, _FillValue=NaN
```

### `reproject_datatree`

Reproject all zones in a DataTree to a common CRS. 

While not recommended, you can provide `dst_nodata` to reproject with a different nodata value.

Generally speaking, the library handles the change in the nodata value, e.g. when you
dequantise from int8 to float32, nodata changes from -128 to NaN.  This is done 
internally by setting the correct nodata value on the output of each transformation via 
`set_aef_nodata` so that downstream tools like `xr_reproject` can read it and handle it correctly during reprojection.

```python
from aef_loader.utils import reproject_datatree
from odc.geo.geobox import GeoBox

target = GeoBox.from_bbox(
    bbox=(-122.5, 37.5, -122.0, 38.0),
    crs="EPSG:4326",
    resolution=0.0001,
)
combined = reproject_datatree(tree, target)
```

### `split_bands`

Split a multi-band dataset into individual band variables.

```python
from aef_loader import split_bands

split = split_bands(dataset)
```
