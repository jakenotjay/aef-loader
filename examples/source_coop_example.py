#!/usr/bin/env python
"""
Example: Using aef-loader with Source Cooperative (AWS S3) hosted data.

This example demonstrates how to use aef-loader with the Source Cooperative
hosted Alpha Earth Foundations (AEF) embeddings, which provides public access
without requiring GCP credentials or requester-pays.

Key Differences from GCS:
  - No authentication required (public bucket)
  - Data stored on AWS S3 instead of Google Cloud Storage
  - 2017 data not available (data quality issue)

Usage:
  # Source Cooperative (no auth needed)
  python examples/source_coop_example.py

  # Run full workflow
  python examples/source_coop_example.py --full-workflow

For more information:
  - Source Cooperative: https://source.coop/tge-labs/aef
  - AEF Documentation: https://developers.google.com/earth-engine/guides/aef_on_gcs_readme
"""

import argparse
import asyncio
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("source_coop_example")


async def demonstrate_source_coop_index():
    """Download and explore the Source Cooperative AEF parquet index."""
    from aef_loader import AEFIndex, DataSource

    logger.info("=== Source Cooperative Index Demo ===")
    logger.info("Note: No GCP credentials required for Source Cooperative")

    try:
        index = AEFIndex(source=DataSource.SOURCE_COOP)
        index_path = await index.download()
        logger.info(f"AEF index downloaded to: {index_path}")

        # Load and explore the index
        gdf = index.load()
        logger.info("\nSource Cooperative AEF Index Summary:")
        logger.info(f"  Total tiles: {len(gdf):,}")
        logger.info(f"  Years available: {sorted(gdf['year'].unique())}")
        logger.info(f"  Columns: {list(gdf.columns)}")

        return gdf
    except Exception as e:
        logger.error(f"Failed to download index: {e}")
        import traceback

        traceback.print_exc()
        return None


async def demonstrate_source_coop_reader():
    """Demonstrate using VirtualTiffReader with Source Cooperative data."""
    from aef_loader import AEFIndex, DataSource, VirtualTiffReader

    logger.info("\n=== Source Cooperative VirtualTiffReader Demo ===")

    try:
        # First get tiles from the index
        index = AEFIndex(source=DataSource.SOURCE_COOP)
        await index.download()
        index.load()

        # Query for tiles - San Francisco area, 2023
        tiles = await index.query(
            bbox=(-122.5, 37.5, -122.0, 38.0),
            years=(2023, 2023),
            limit=5,
        )

        if not tiles:
            logger.warning("No tiles found for the query")
            return None

        logger.info(f"Found {len(tiles)} tiles")
        for tile in tiles[:3]:
            logger.info(f"  - {tile.id}: year={tile.year}, zone={tile.utm_zone}")

        # Open tiles organized by UTM zone
        logger.info("\nOpening tiles by UTM zone...")
        async with VirtualTiffReader() as reader:
            tree = await reader.open_tiles_by_zone(tiles)

            logger.info(f"\nDataTree structure: {list(tree.children.keys())}")
            for zone in tree.children:
                ds = tree[zone].ds
                logger.info(f"\n  Zone {zone}:")
                logger.info(f"    CRS: {ds.odc.crs}")
                logger.info(f"    Dimensions: {dict(ds.sizes)}")
                logger.info(
                    f"    Data variables: {list(ds.data_vars)[:5]}... ({len(ds.data_vars)} total)"
                )
                logger.info(f"    Coordinates: {list(ds.coords)}")

            return tree

    except Exception as e:
        logger.error(f"Failed to open COGs: {e}")
        import traceback

        traceback.print_exc()
        return None


async def run_source_coop_workflow(
    limit: int = 5,
    output_geojson: Path | None = None,
):
    """
    Run a complete aef-loader workflow with Source Cooperative data.

    This demonstrates:
    1. Querying the Source Cooperative AEF index
    2. Opening tiles organized by UTM zone via virtual-tiff
    3. Filtering to a complex geometry
    4. Reprojecting to a common CRS
    5. Dequantizing embeddings
    """
    import json

    from aef_loader import (
        AEFIndex,
        DataSource,
        VirtualTiffReader,
        clip_to_geometry,
        dequantize_aef,
        reproject_datatree,
    )
    from odc.geo.geobox import GeoBox

    logger.info("\n=== Source Cooperative Embeddings Workflow ===")

    # Define area of interest - San Francisco area
    bbox = (-122.5, 37.7, -122.3, 37.9)
    years = (2023, 2023)

    logger.info(f"Area of interest: {bbox}")
    logger.info(f"Years: {years}")
    logger.info(f"Processing limit: {limit} tiles")
    logger.info("Data source: Source Cooperative (S3)")

    try:
        # Step 1: Query index
        logger.info("\nStep 1: Querying Source Cooperative AEF index...")
        index = AEFIndex(source=DataSource.SOURCE_COOP)
        await index.download()
        index.load()

        tiles = await index.query(
            bbox=bbox,
            years=years,
            limit=limit,
        )

        if not tiles:
            logger.warning("No tiles found for the specified area and time range")
            return None

        logger.info(f"Found {len(tiles)} tiles")
        for tile in tiles[:3]:
            logger.info(f"  - {tile.id}: year={tile.year}, zone={tile.utm_zone}")

        # Optionally output queried tiles as GeoJSON
        if output_geojson:
            features = []
            for tile in tiles:
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "id": tile.id,
                            "year": tile.year,
                            "utm_zone": tile.utm_zone,
                            "crs_epsg": tile.crs_epsg,
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [tile.bbox[0], tile.bbox[1]],
                                    [tile.bbox[2], tile.bbox[1]],
                                    [tile.bbox[2], tile.bbox[3]],
                                    [tile.bbox[0], tile.bbox[3]],
                                    [tile.bbox[0], tile.bbox[1]],
                                ]
                            ],
                        },
                    }
                )
            geojson = {"type": "FeatureCollection", "features": features}
            output_geojson.write_text(json.dumps(geojson, indent=2))
            logger.info(f"Saved queried tiles to {output_geojson}")

        # Step 2: Load tiles by UTM zone
        logger.info("\nStep 2: Loading tiles by UTM zone...")
        async with VirtualTiffReader() as reader:
            tree = await reader.open_tiles_by_zone(tiles)

        logger.info(f"Loaded DataTree with zones: {list(tree.children.keys())}")
        for zone in tree.children:
            ds = tree[zone].ds
            logger.info(f"  Zone {zone}: {dict(ds.sizes)}")

        # Step 3: Demonstrate geometry clipping (optional)
        clip_geometry = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-122.45, 37.75],
                    [-122.35, 37.75],
                    [-122.35, 37.85],
                    [-122.45, 37.85],
                    [-122.45, 37.75],
                ]
            ],
        }
        logger.info("\nStep 3: Clipping to geometry...")
        for zone in list(tree.children.keys()):
            ds = tree[zone].ds
            clipped = clip_to_geometry(ds, clip_geometry)
            logger.info(f"  Zone {zone} after clip: {dict(clipped.sizes)}")
            # Update tree with clipped data
            tree[zone].ds = clipped

        # Step 4: Reproject to common CRS
        logger.info("\nStep 4: Reprojecting to common CRS...")
        target_geobox = GeoBox.from_bbox(
            bbox=bbox,
            crs="EPSG:4326",
            resolution=0.0001,  # ~10m at this latitude
        )
        combined = reproject_datatree(tree, target_geobox)
        logger.info(f"Combined dataset: {dict(combined.sizes)}")
        logger.info(f"CRS: {combined.odc.crs}")

        # Step 5: Dequantize embeddings
        logger.info("\nStep 5: Dequantizing first band (A00)...")
        a00 = combined["A00"]
        a00_dequant = dequantize_aef(a00)
        logger.info(f"  Original dtype: {a00.dtype}")
        logger.info(f"  Dequantized dtype: {a00_dequant.dtype}")
        logger.info(f"  Shape: {a00_dequant.shape}")

        logger.info("\nWorkflow complete!")
        return combined

    except Exception as e:
        logger.error(f"Failed to run workflow: {e}")
        import traceback

        traceback.print_exc()
        return None


async def main(
    full_workflow: bool = False,
    limit: int = 5,
    output_geojson: Path | None = None,
):
    """Run Source Cooperative demonstrations."""
    logger.info("aef-loader Source Cooperative Example")
    logger.info("=" * 60)
    logger.info("\nSource Cooperative provides public access to AEF embeddings")
    logger.info("hosted on AWS S3, without requiring GCP credentials.")
    logger.info("\nKey features:")
    logger.info("  - No authentication required")
    logger.info("  - Same data as GCS (except 2017)")

    # Demo 1: Index download (always run)
    await demonstrate_source_coop_index()

    # Demo 2: VirtualTiffReader with zone-based loading
    if full_workflow:
        await demonstrate_source_coop_reader()

    # Demo 3: Full workflow
    if full_workflow:
        await run_source_coop_workflow(limit=limit, output_geojson=output_geojson)

    logger.info("\n" + "=" * 60)
    logger.info("Source Cooperative demonstration complete!")

    if not full_workflow:
        logger.info(
            "\nTo run the full workflow:\n"
            "  python examples/source_coop_example.py --full-workflow"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate aef-loader with Source Cooperative (AWS S3)"
    )
    parser.add_argument(
        "--full-workflow",
        action="store_true",
        help="Run the complete embedding retrieval workflow",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of tiles to process (default: 5)",
    )
    parser.add_argument(
        "--output-geojson",
        type=Path,
        help="Output path for queried tiles GeoJSON",
        default=None,
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            full_workflow=args.full_workflow,
            limit=args.limit,
            output_geojson=args.output_geojson,
        )
    )
