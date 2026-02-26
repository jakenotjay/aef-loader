#!/usr/bin/env python
"""
Example: Using aef-loader with Alpha Earth Foundations (AEF) Embeddings on GCS

This example demonstrates how to use aef-loader to access AEF 64-channel
satellite embeddings stored on Google Cloud Storage (requester-pays).

For public access without GCP credentials, use Source Cooperative instead:
  python examples/source_coop_example.py

Prerequisites:
  1. Google Cloud credentials: gcloud auth application-default login
  2. A GCP project with billing enabled (AEF is a requester-pays dataset)
  3. Set your project: export GOOGLE_CLOUD_PROJECT=your-project-id

Usage:
  python examples/aef_embeddings_example.py --project your-gcp-project-id
  python examples/aef_embeddings_example.py --project your-gcp-project-id --full-workflow

For more information about AEF:
  https://developers.google.com/earth-engine/guides/aef_on_gcs_readme
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("aef_example")


async def demonstrate_aef_index_download(project: str | None = None):
    """Download and explore the AEF parquet index."""
    from aef_loader import AEFIndex, DataSource

    logger.info("=== AEF Index Download Demo ===")

    if not project:
        logger.warning(
            "GCP project required to download AEF index (requester-pays).\n"
            "Set --project or GOOGLE_CLOUD_PROJECT environment variable."
        )
        return None

    try:
        index = AEFIndex(source=DataSource.GCS, gcp_project=project)
        index_path = await index.download()
        logger.info(f"AEF index downloaded to: {index_path}")

        # Load and explore the index
        gdf = index.load()
        logger.info("\nAEF Index Summary:")
        logger.info(f"  Total tiles: {len(gdf):,}")
        logger.info(f"  Years available: {sorted(gdf['year'].unique())}")
        logger.info(f"  Columns: {list(gdf.columns)}")

        return gdf
    except Exception as e:
        logger.error(f"Failed to download index: {e}")
        return None


async def demonstrate_virtual_tiff_reader(project: str | None = None):
    """Demonstrate using VirtualTiffReader with zone-based loading."""
    from aef_loader import AEFIndex, DataSource, VirtualTiffReader

    logger.info("\n=== VirtualTiffReader Demo ===")

    if not project:
        logger.warning("Skipping VirtualTiffReader demo - no project specified")
        return None

    try:
        # First get tiles from the index
        index = AEFIndex(source=DataSource.GCS, gcp_project=project)
        await index.download()
        index.load()

        # Query for tiles
        tiles = await index.query(
            bbox=(-122.5, 37.5, -122.0, 38.0),  # San Francisco area
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
        async with VirtualTiffReader(gcp_project=project) as reader:
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

            return tree

    except Exception as e:
        logger.error(f"Failed to open COGs: {e}")
        import traceback

        traceback.print_exc()
        return None


async def run_aef_workflow(
    project: str | None = None,
    limit: int = 5,
    output_geojson: Path | None = None,
):
    """
    Run a complete aef-loader workflow with AEF embeddings.

    This demonstrates:
    1. Querying the AEF parquet index
    2. Opening tiles organized by UTM zone via virtual-tiff
    3. Reprojecting to a common CRS
    4. Dequantizing embeddings
    """
    import json

    from aef_loader import (
        AEFIndex,
        DataSource,
        VirtualTiffReader,
        dequantize_aef,
    )
    from aef_loader.utils import reproject_datatree
    from odc.geo.geobox import GeoBox

    logger.info("\n=== AEF Embeddings Workflow ===")

    if not project:
        logger.warning(
            "Skipping AEF workflow - no project specified.\n"
            "To run the workflow, provide --project YOUR_PROJECT_ID"
        )
        return None

    # Define area of interest - San Francisco area
    bbox = (-122.5, 37.7, -122.3, 37.9)
    years = (2023, 2023)

    logger.info(f"Area of interest: {bbox}")
    logger.info(f"Years: {years}")
    logger.info(f"Processing limit: {limit} tiles")

    try:
        # Step 1: Query index
        logger.info("\nStep 1: Querying AEF index...")
        index = AEFIndex(source=DataSource.GCS, gcp_project=project)
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
        async with VirtualTiffReader(gcp_project=project) as reader:
            tree = await reader.open_tiles_by_zone(tiles)

        logger.info(f"Loaded DataTree with zones: {list(tree.children.keys())}")
        for zone in tree.children:
            ds = tree[zone].ds
            logger.info(f"  Zone {zone}: {dict(ds.sizes)}")

        # Step 3: Reproject to common CRS
        logger.info("\nStep 3: Reprojecting to common CRS...")
        target_geobox = GeoBox.from_bbox(
            bbox=bbox,
            crs="EPSG:4326",
            resolution=0.0001,  # ~10m at this latitude
        )
        combined = reproject_datatree(tree, target_geobox)
        logger.info(f"Combined dataset: {dict(combined.sizes)}")
        logger.info(f"CRS: {combined.odc.crs}")

        # Step 4: Dequantize embeddings
        logger.info("\nStep 4: Dequantizing first band (A00)...")
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
    project: str | None = None,
    full_workflow: bool = False,
    limit: int = 5,
    output_geojson: Path | None = None,
):
    """Run AEF demonstrations."""
    logger.info("aef-loader Alpha Earth Foundations (AEF) Example")
    logger.info("=" * 60)
    logger.info("\nAEF provides 64-channel satellite embeddings for ML applications.")
    logger.info("Data is stored on GCS as requester-pays.")
    logger.info(
        "\nUsing virtual-tiff for efficient COG access without data duplication."
    )
    logger.info(
        "\nNote: For public access without GCP credentials, use Source Cooperative:"
    )
    logger.info("  python examples/source_coop_example.py")

    # Demo 1: Index download
    if project:
        await demonstrate_aef_index_download(project)
    else:
        logger.info("\n=== Skipping Index Download (no project specified) ===")

    # Demo 2: VirtualTiffReader
    if project and full_workflow:
        await demonstrate_virtual_tiff_reader(project)

    # Demo 3: Full workflow
    if full_workflow:
        if project:
            try:
                ds = await run_aef_workflow(
                    project, limit=limit, output_geojson=output_geojson
                )
                if ds is not None:
                    logger.info("\nAEF workflow completed successfully!")
            except Exception as e:
                logger.error(f"Workflow failed: {e}")
        else:
            logger.warning("\nSkipping full workflow - no GCP project specified")

    logger.info("\n" + "=" * 60)
    logger.info("AEF demonstration complete!")

    if not project:
        logger.info(
            "\nTo run with actual data access:\n"
            "  1. Run: gcloud auth application-default login\n"
            "  2. Run: python examples/aef_embeddings_example.py --project YOUR_PROJECT\n"
            "  3. Add --full-workflow to run the complete embedding retrieval\n"
            "  4. Use --limit N to process only N tiles (default: 5)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate aef-loader with AEF embeddings on GCS"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="GCP project ID for requester-pays access",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
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

    asyncio.run(main(args.project, args.full_workflow, args.limit, args.output_geojson))
