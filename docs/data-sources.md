# Data Sources

aef-loader supports two dataset hosts, each with tradeoffs.

## Source Cooperative

- **No authentication required** — Public bucket on AWS S3
- **URL**: `s3://us-west-2.opendata.source.coop/tge-labs/aef/`
- **Limitations**: 2017 and 2025 data not yet available
- **Documentation**: https://source.coop/tge-labs/aef

```python
from aef_loader import AEFIndex, DataSource

index = AEFIndex(source=DataSource.SOURCE_COOP)
```

## Google Cloud Storage

- **Requires GCP credentials** — Requester-pays bucket
- **URL**: `gs://alphaearth_foundations/`
- **Maintained by the Earth Engine team** — Most up to date
- **Documentation**: https://developers.google.com/earth-engine/guides/aef_on_gcs_readme

```python
index = AEFIndex(source=DataSource.GCS, gcp_project="my-project")
```

!!! warning
    GCS uses [requester-pays](https://docs.google.com/storage/docs/requester-pays) billing. You will be charged for egress and API requests against your GCP project.

### Authentication
Providing a gcp_project itself alone is not enough you'll need to also create 
[application-default credentials](https://docs.cloud.google.com/docs/authentication/application-default-credentials).