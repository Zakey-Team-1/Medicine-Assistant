"""Tests for the configuration module."""

import os
from unittest.mock import patch

import pytest

from medicine_assistant.config import Settings


class TestSettings:
    """Test cases for the Settings class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        assert Settings.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
        assert Settings.MODEL_NAME == "openai/gpt-4o-mini" or os.getenv("MODEL_NAME")
        assert Settings.CHUNK_SIZE == 1000 or os.getenv("CHUNK_SIZE")
        assert Settings.CHUNK_OVERLAP == 200 or os.getenv("CHUNK_OVERLAP")
        assert Settings.TOP_K_RESULTS == 5 or os.getenv("TOP_K_RESULTS")

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
