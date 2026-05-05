"""Forest Data Partnership (FDP) commodity-probability accessors.

Sibling to the AEF accessors. Reuses the shared cloud helpers in
``aef_loader._cloud`` and follows the same ``download/build → load → query``
shape as :class:`aef_loader.AEFIndex`, but builds its catalogue locally by
listing the GCS bucket since the FDP project does not publish a parquet index.
"""

from aef_loader.fdp.index import FDPIndex

__all__ = ["FDPIndex"]
