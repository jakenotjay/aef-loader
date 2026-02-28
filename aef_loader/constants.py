"""Constants for AEF loader."""

from enum import Enum


class DataSource(Enum):
    """Data source for AEF embeddings."""

    GCS = "gcs"
    SOURCE_COOP = "source_coop"


# GCS (Google Cloud Storage) configuration - original source
GCS_BUCKET = "alphaearth_foundations"
GCS_INDEX_BLOB = "satellite_embedding/v1/annual/aef_index.parquet"
GCS_INDEX_URL = f"gs://{GCS_BUCKET}/{GCS_INDEX_BLOB}"
GCS_BASE_PATH = f"gs://{GCS_BUCKET}/satellite_embedding/v1/annual"

# Source Cooperative (AWS S3) configuration - alternative source
# Data is hosted at s3://us-west-2.opendata.source.coop/tge-labs/aef/...
SOURCE_COOP_BUCKET = "us-west-2.opendata.source.coop"
SOURCE_COOP_REGION = "us-west-2"
SOURCE_COOP_PREFIX = "tge-labs/aef"
SOURCE_COOP_INDEX_BLOB = f"{SOURCE_COOP_PREFIX}/v1/annual/aef_index.parquet"
SOURCE_COOP_INDEX_URL = f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_INDEX_BLOB}"
SOURCE_COOP_BASE_PATH = f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_PREFIX}/v1/annual"

# AEF data characteristics
AEF_NUM_CHANNELS = 64
AEF_CHANNEL_NAMES = [f"A{i:02d}" for i in range(AEF_NUM_CHANNELS)]

# Dequantization parameters for AEF embeddings
# Embeddings are stored as int8 and dequantized using: ((v/127.5)² × sign(v))
AEF_DEQUANT_DIVISOR = 127.5
AEF_NODATA_VALUE = -128
