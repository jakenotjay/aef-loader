"""Tests for aef_loader.reader module."""

from unittest.mock import MagicMock, patch

import pytest
from aef_loader.reader import (
    VirtualTiffReader,
    _normalize_band_indices,
    _parse_gcs_path,
)


class TestParseGcsPath:
    """Tests for _parse_gcs_path helper."""

    @pytest.mark.unit
    def test_parse_gs_url(self):
        """Test parsing gs:// URL."""
        bucket, key = _parse_gcs_path("gs://my-bucket/path/to/file.tif")

        assert bucket == "my-bucket"
        assert key == "path/to/file.tif"


class TestVirtualTiffReader:
    """Tests for VirtualTiffReader class."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with VirtualTiffReader() as reader:
            assert reader is not None

        # After exit, stores should be cleared
        assert reader._stores == {}

    @pytest.mark.unit
    def test_get_gcs_store_with_project(self):
        """Test that GCS store includes requester-pays header when project is set."""
        reader = VirtualTiffReader(gcp_project="my-project")
        with patch("aef_loader.reader.GCSStore") as mock_gcs:
            reader._get_gcs_store("my-bucket")
            mock_gcs.assert_called_once_with(
                bucket="my-bucket",
                client_options={
                    "default_headers": {"x-goog-user-project": "my-project"}
                },
            )

    @pytest.mark.unit
    def test_get_gcs_store_without_project_raises(self):
        """Test that GCS store raises when no project is set."""
        reader = VirtualTiffReader()
        with pytest.raises(ValueError, match="gcp_project is required"):
            reader._get_gcs_store("my-bucket")

    @pytest.mark.unit
    def test_get_store_caches(self):
        """Test that _get_store caches stores."""
        reader = VirtualTiffReader()
        mock_store = MagicMock()
        reader._stores["gs://my-bucket"] = mock_store

        with patch("obstore.store.GCSStore") as mock_gcs:
            store = reader._get_store("gs", "my-bucket")

            mock_gcs.assert_not_called()
            assert store == mock_store


class TestNormalizeBandIndices:
    """Tests for _normalize_band_indices — called directly, no mocking needed."""

    @pytest.mark.unit
    def test_string_names(self):
        assert _normalize_band_indices(["A00", "A03", "A63"]) == [0, 3, 63]

    @pytest.mark.unit
    def test_int_indices(self):
        assert _normalize_band_indices([0, 3, 63]) == [0, 3, 63]

    @pytest.mark.unit
    def test_single_band(self):
        assert _normalize_band_indices(["A32"]) == [32]

    @pytest.mark.unit
    def test_invalid_name_format(self):
        with pytest.raises(ValueError, match="Invalid band name"):
            _normalize_band_indices(["X00"])

    @pytest.mark.unit
    def test_invalid_name_too_short(self):
        with pytest.raises(ValueError, match="Invalid band name"):
            _normalize_band_indices(["A0"])

    @pytest.mark.unit
    def test_out_of_range_name(self):
        with pytest.raises(ValueError, match="out of range"):
            _normalize_band_indices(["A64"])

    @pytest.mark.unit
    def test_out_of_range_int(self):
        with pytest.raises(ValueError, match="out of range"):
            _normalize_band_indices([64])

    @pytest.mark.unit
    def test_negative_int(self):
        with pytest.raises(ValueError, match="out of range"):
            _normalize_band_indices([-1])

    @pytest.mark.unit
    def test_mixed_types_str_first(self):
        with pytest.raises(TypeError, match="not a mix"):
            _normalize_band_indices(["A00", 1])  # type: ignore[list-item]

    @pytest.mark.unit
    def test_mixed_types_int_first(self):
        with pytest.raises(TypeError, match="not a mix"):
            _normalize_band_indices([0, "A01"])  # type: ignore[list-item]

    @pytest.mark.unit
    def test_empty_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            _normalize_band_indices([])

    @pytest.mark.unit
    def test_preserves_order(self):
        assert _normalize_band_indices(["A63", "A00", "A32"]) == [63, 0, 32]
