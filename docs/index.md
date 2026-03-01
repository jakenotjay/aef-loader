# aef-loader

Virtualizarr access for AEF embeddings as an analysis ready data cube, alongside rapid querying of the GCS and Source Coop index. 2x quicker than rioxarray for single tile downloads.

## What is AEF?

Alpha Earth Foundations embeddings is a dataset produced by Google Deepmind, providing a yearly 64-channel embeddings derived from numerous satellite image sources with numerous downstream applications. The embeddings are stored as multi-band Cloud-Optimised GeoTIFFs (COGs), alongside a parquet index file.

AEF is stored by two hosts:

- Google Cloud Storage (official support) - requester pays (requires gcp_project)
- Source Cooperative - AWS hosted and free to access

More in the docs.

## What does aef-loader do?

aef-loader provides two key functionalities:

1. Rapid download, and querying of indexes for source_coop + gcs with obstore and geopandas
2. Lazily load the COGs as VirtualiZarr as a datatree by UTM zone, COG headers are cached, so repeated reads are cheap(er)

As additional utilities:

- dequantize, and requantize the embeddings
- split the "embeddings" dataset into 64 datasets
- use odc-geobox for dask aware reprojections for creating multi-zone composites

## Overview

Alpha Earth Foundations embeddings is a dataset produced by Google Deepmind, providing a yearly 64-channel embeddings derived from numerous satellite image sources with numerous downstream applications. The embeddings are stored as multi-band Cloud-Optimised GeoTIFFs (COGs).

aef-loader supports two dataset hosts, both having tradeoffs:

1. [Google Cloud Storage](https://developers.google.com/earth-engine/guides/aef_on_gcs_readme) - maintained by the [Earth Engine](https://earthengine.google.com/) team, more up to date but requiring authentication and "[requester pays](https://docs.cloud.google.com/storage/docs/requester-pays)", meaning users must pay egress and other charges.
2. [Source Cooperative](https://source.coop/tge-labs/aef) - Hosted on AWS S3 and free to access, but generally less up to date (currently missing 2017 and 2025)


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
    index.load() # returns a gdf for alternative use

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
        resolution=0.0001,
    )
    combined = reproject_datatree(tree, target)

asyncio.run(main())
```

## Attribution and Dataset License
This dataset is licensed under CC-BY 4.0 and requires the following attribution text: "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."

## Special notes
Thanks to Max Jones, [Virtual-tiff](https://github.com/virtual-zarr/virtual-tiff) and [Virtualizarr](https://github.com/zarr-developers/VirtualiZarr).