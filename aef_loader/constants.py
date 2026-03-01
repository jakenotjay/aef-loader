"""Constants for AEF loader."""

from enum import Enum


class DataSource(Enum):
    """Data source for AEF embeddings."""

    GCS = "gcs"
    SOURCE_COOP = "source_coop"


# GCS (Google Cloud Storage) configuration - original source
GCS_BUCKET = "alphaearth_foundations"
GCS_INDEX_BLOB = "satellite_embedding/v1/annual/aef_index.parquet"

# Source Cooperative (AWS S3) configuration - alternative source
# Data is hosted at s3://us-west-2.opendata.source.coop/tge-labs/aef/...
SOURCE_COOP_BUCKET = "us-west-2.opendata.source.coop"
SOURCE_COOP_REGION = "us-west-2"
SOURCE_COOP_INDEX_BLOB = "tge-labs/aef/v1/annual/aef_index.parquet"

# Dequantization parameters for AEF embeddings
# Embeddings are stored as int8 and dequantized using: ((v/127.5)² × sign(v))
AEF_DEQUANT_DIVISOR = 127.5
AEF_NODATA_VALUE = -128
