"""Tests for aef_loader.fdp.index module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from aef_loader.fdp import FDPIndex
from aef_loader.types import FDPTileInfo


class TestFDPIndex:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_build_requires_project(self, tmp_path):
        index = FDPIndex(release="2025b", cache_dir=tmp_path)
        with pytest.raises(ValueError, match="gcp_project is required"):
            await index.build()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_build_uses_cache(self, tmp_path):
        cache_file = tmp_path / "fdp_index_2025b.parquet"
        cache_file.touch()

        index = FDPIndex(release="2025b", gcp_project="test", cache_dir=tmp_path)
        result = await index.build()

        assert result == cache_file

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_build_force_rebuilds_even_if_cached(self, tmp_path, mock_fdp_gdf):
        cache_file = tmp_path / "fdp_index_2025b.parquet"
        cache_file.write_bytes(b"stale")

        # Patch obstore listings to return a controlled hierarchy.
        with (
            patch(
                "aef_loader.fdp.index.obs.list_with_delimiter_async",
                new_callable=AsyncMock,
            ) as mock_delim,
            patch("aef_loader.fdp.index.obs.list") as mock_list,
        ):
            # Two calls: discover commodities, then years per commodity.
            mock_delim.side_effect = [
                {
                    "common_prefixes": [
                        "forestdatapartnership/2025b/coffee",
                        "forestdatapartnership/2025b/palm_2023",  # filtered out
                    ],
                    "objects": [],
                },
                {
                    "common_prefixes": [
                        "forestdatapartnership/2025b/coffee/2024",
                    ],
                    "objects": [],
                },
            ]

            class _AsyncIter:
                def __init__(self, batches):
                    self._batches = batches

                def __aiter__(self):
                    return self._iter()

                async def _iter(self):
                    for b in self._batches:
                        yield b

            mock_list.return_value = _AsyncIter(
                [
                    [
                        {
                            "path": "forestdatapartnership/2025b/coffee/2024/lng_9_lat_5.tif",
                            "size": 100,
                        },
                        {
                            "path": "forestdatapartnership/2025b/coffee/2024/lng_10_lat_5.tif",
                            "size": 100,
                        },
                        {
                            "path": "forestdatapartnership/2025b/coffee/2024",
                            "size": 0,
                        },  # directory marker
                        # Non-tile file: should be silently skipped (warning
                        # logged) and excluded from the resulting parquet.
                        {
                            "path": "forestdatapartnership/2025b/coffee/2024/README.txt",
                            "size": 42,
                        },
                    ]
                ]
            )

            index = FDPIndex(release="2025b", gcp_project="test", cache_dir=tmp_path)
            result = await index.build(force=True)

        assert result == cache_file
        # Re-loaded fresh — original "stale" bytes overwritten.
        gdf = index.load()
        assert len(gdf) == 2
        assert set(gdf["commodity"]) == {"coffee"}
        assert set(zip(gdf["lng"], gdf["lat"])) == {(9, 5), (10, 5)}
        # Non-matching file is excluded.
        assert not gdf["path"].str.endswith("README.txt").any()

    @pytest.mark.unit
    def test_load_not_found_raises(self, tmp_path):
        index = FDPIndex(release="2025b", cache_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            index.load()

    @pytest.mark.unit
    def test_load_with_mock_gdf(self, tmp_path, mock_fdp_gdf):
        cache_file = tmp_path / "fdp_index_2025b.parquet"
        mock_fdp_gdf.to_parquet(cache_file)

        index = FDPIndex(release="2025b", cache_dir=tmp_path)
        gdf = index.load()
        assert len(gdf) == len(mock_fdp_gdf)
        assert {"commodity", "year", "lng", "lat", "path", "geometry"} <= set(
            gdf.columns
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_filters_by_bbox_year_commodity(self, tmp_path, mock_fdp_gdf):
        cache_file = tmp_path / "fdp_index_2025b.parquet"
        mock_fdp_gdf.to_parquet(cache_file)

        index = FDPIndex(release="2025b", cache_dir=tmp_path)

        # bbox strictly inside the (lng=9, lat=5) tile — avoids edge-touching
        # neighbours under shapely's intersects semantics.
        tiles = await index.query(
            bbox=(9.1, 5.1, 9.9, 5.9),
            years=2024,
            commodities=["coffee"],
        )
        assert len(tiles) == 1
        t = tiles[0]
        assert isinstance(t, FDPTileInfo)
        assert t.commodity == "coffee"
        assert t.year == 2024
        assert t.lng == 9 and t.lat == 5
        assert t.bbox == (9.0, 5.0, 10.0, 6.0)  # always full 1° tile
        assert t.crs_epsg == 4326

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_year_range(self, tmp_path, mock_fdp_gdf):
        cache_file = tmp_path / "fdp_index_2025b.parquet"
        mock_fdp_gdf.to_parquet(cache_file)

        index = FDPIndex(release="2025b", cache_dir=tmp_path)
        tiles = await index.query(years=(2020, 2024), commodities=["coffee"])
        # 3 spatial tiles × 2 years = 6
        assert len(tiles) == 6
        assert {t.year for t in tiles} == {2020, 2024}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_limit(self, tmp_path, mock_fdp_gdf):
        cache_file = tmp_path / "fdp_index_2025b.parquet"
        mock_fdp_gdf.to_parquet(cache_file)

        index = FDPIndex(release="2025b", cache_dir=tmp_path)
        tiles = await index.query(limit=2)
        assert len(tiles) == 2

    @pytest.mark.unit
    def test_default_cache_dir_without_xdg(self, monkeypatch):
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        index = FDPIndex(release="2025b")
        assert index.cache_dir == Path.home() / ".cache" / "aef-loader"

    @pytest.mark.unit
    def test_default_cache_dir_honours_xdg(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        index = FDPIndex(release="2025b")
        assert index.cache_dir == tmp_path / "aef-loader"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_limit_zero_returns_empty(self, tmp_path, mock_fdp_gdf):
        cache_file = tmp_path / "fdp_index_2025b.parquet"
        mock_fdp_gdf.to_parquet(cache_file)

        index = FDPIndex(release="2025b", cache_dir=tmp_path)
        tiles = await index.query(limit=0)
        assert tiles == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_accepts_string_year(self, tmp_path, mock_fdp_gdf):
        cache_file = tmp_path / "fdp_index_2025b.parquet"
        mock_fdp_gdf.to_parquet(cache_file)

        index = FDPIndex(release="2025b", cache_dir=tmp_path)
        tiles = await index.query(years="2024", commodities=["coffee"])
        assert {t.year for t in tiles} == {2024}
        assert len(tiles) == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_build_collects_rows_across_multiple_batches(self, tmp_path):
        """obs.list yields rows in batches; the index must concatenate them."""
        with (
            patch(
                "aef_loader.fdp.index.obs.list_with_delimiter_async",
                new_callable=AsyncMock,
            ) as mock_delim,
            patch("aef_loader.fdp.index.obs.list") as mock_list,
        ):
            mock_delim.side_effect = [
                {
                    "common_prefixes": ["forestdatapartnership/2025b/coffee"],
                    "objects": [],
                },
                {
                    "common_prefixes": ["forestdatapartnership/2025b/coffee/2024"],
                    "objects": [],
                },
            ]

            class _AsyncIter:
                def __init__(self, batches):
                    self._batches = batches

                def __aiter__(self):
                    return self._iter()

                async def _iter(self):
                    for b in self._batches:
                        yield b

            mock_list.return_value = _AsyncIter(
                [
                    [
                        {
                            "path": "forestdatapartnership/2025b/coffee/2024/lng_9_lat_5.tif",
                            "size": 100,
                        }
                    ],
                    [
                        {
                            "path": "forestdatapartnership/2025b/coffee/2024/lng_10_lat_5.tif",
                            "size": 100,
                        }
                    ],
                ]
            )

            index = FDPIndex(release="2025b", gcp_project="test", cache_dir=tmp_path)
            await index.build(force=True)

        gdf = index.load()
        assert set(zip(gdf["lng"], gdf["lat"])) == {(9, 5), (10, 5)}

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "years,expected",
        [
            (2024, (2024, 2024)),
            ("2024", (2024, 2024)),
            ("2024-06-01", (2024, 2024)),
            ((2020, 2024), (2020, 2024)),
            (("2020", "2024"), (2020, 2024)),
            ((2020, "2024-12-31"), (2020, 2024)),
            # Pin current pass-through behaviour for inverted ranges. Callers
            # currently get an empty result silently — we can tighten the
            # contract (raise / clamp) later if it becomes a footgun.
            ((2024, 2020), (2024, 2020)),
        ],
    )
    def test_year_range_normalises_inputs(self, years, expected):
        assert FDPIndex._year_range(years) == expected
