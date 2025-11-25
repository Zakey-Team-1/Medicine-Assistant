"""Tests for the configuration module."""

from unittest.mock import patch

import pytest

from src.config import Settings


class TestSettings:
    """Test cases for the Settings class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        assert Settings.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
        # Check that settings have expected defaults or are set from environment
        assert Settings.MODEL_NAME is not None
        assert Settings.CHUNK_SIZE > 0
        assert Settings.CHUNK_OVERLAP >= 0
        assert Settings.TOP_K_RESULTS > 0

    def test_validate_missing_api_key(self):
        """Test that validation fails without API key."""
        with patch.object(Settings, "OPENROUTER_API_KEY", ""):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                Settings.validate()

    def test_validate_with_api_key(self):
        """Test that validation passes with API key."""
        with patch.object(Settings, "OPENROUTER_API_KEY", "test-key"):
            # Should not raise
            Settings.validate()
