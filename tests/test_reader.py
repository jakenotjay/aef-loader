"""Tests for aef_loader.reader module."""

from unittest.mock import MagicMock, patch

import pytest
from aef_loader._cloud import parse_gcs_path as _parse_gcs_path
from aef_loader.reader import VirtualTiffReader


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
    def test_get_store_passes_project_to_make_gcs_store(self):
        """GCS store creation should forward the requester-pays project."""
        reader = VirtualTiffReader(gcp_project="my-project")
        with patch("aef_loader.reader.make_gcs_store") as mock_make:
            reader._get_store("gs", "my-bucket")
            mock_make.assert_called_once_with("my-bucket", "my-project")

    @pytest.mark.unit
    def test_get_store_without_project_raises(self):
        """GCS store creation must require a project for requester-pays."""
        reader = VirtualTiffReader()
        with pytest.raises(ValueError, match="gcp_project is required"):
            reader._get_store("gs", "my-bucket")

    @pytest.mark.unit
    def test_get_store_caches(self):
        """Test that _get_store caches stores."""
        reader = VirtualTiffReader()
        mock_store = MagicMock()
        reader._stores["gs://my-bucket"] = mock_store

        # Patch the symbol the reader actually calls. Patching
        # obstore.store.GCSStore would be vacuous here because the AEF reader
        # delegates through aef_loader.reader.make_gcs_store after the _cloud
        # refactor.
        with patch("aef_loader.reader.make_gcs_store") as mock_make:
            store = reader._get_store("gs", "my-bucket")
            mock_make.assert_not_called()
        assert store is mock_store
