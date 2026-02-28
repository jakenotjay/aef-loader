"""
AEF Loader - Efficient loader for Alpha Earth Foundations embeddings.

Uses virtual-tiff to create virtual zarr stores from COGs,
enabling lazy xarray/dask operations without data duplication.

Supports both Google Cloud Storage (GCS) and Source Cooperative (AWS S3) backends.

The primary access pattern is loading tiles by UTM zone using VirtualTiffReader.open_tiles_by_zone().
For combining data across zones, use reproject_datatree() from the utils module.
"""

from aef_loader.constants import DataSource
from aef_loader.index import AEFIndex
from aef_loader.reader import VirtualTiffReader
from aef_loader.types import AEFTileInfo
from aef_loader.utils import (
    dequantize_aef,
    mask_nodata,
    quantize_aef,
    reproject_datatree,
    split_bands,
)

__all__ = [
    # Core classes
    "AEFIndex",
    "VirtualTiffReader",
    # Types
    "AEFTileInfo",
    "DataSource",
    # Utility functions
    "dequantize_aef",
    "mask_nodata",
    "quantize_aef",
    "reproject_datatree",
    "split_bands",
]

__version__ = "0.1.0"
