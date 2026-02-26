"""Tests for aef_loader.types module."""

import datetime as dt

import pytest


class TestAEFTileInfo:
    """Tests for AEFTileInfo dataclass."""

    @pytest.mark.unit
    def test_as_datetime_property(self, sample_tile_info):
        """Test as_datetime property returns January 1st of year."""
        result = sample_tile_info.as_datetime

        assert result == dt.datetime(2023, 1, 1)

    @pytest.mark.unit
    def test_bbox_tuple(self, sample_tile_info):
        """Test bbox is a tuple of 4 floats."""
        bbox = sample_tile_info.bbox

        assert len(bbox) == 4
        assert all(isinstance(x, float) for x in bbox)
