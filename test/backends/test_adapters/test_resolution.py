"""Tests for the adapter resolution system."""

import pytest

from mellea.backends.adapters.catalog import AdapterStatus, AdapterType
from mellea.backends.adapters.registry import (
    AdapterRepoRegistry,
    AdapterSource,
    HuggingFaceAdapterRepo,
)
from mellea.backends.adapters.resolution import (
    AdapterImplementation,
    AdapterResolutionTable,
)


class TestAdapterType:
    """Test AdapterType helper methods."""

    def test_all_types(self):
        """Test all_types() returns all adapter types."""
        types = AdapterType.all_types()
        assert len(types) == 2
        assert AdapterType.LORA in types
        assert AdapterType.ALORA in types

    def test_from_string(self):
        """Test from_string() conversion."""
        assert AdapterType.from_string("lora") == AdapterType.LORA
        assert AdapterType.from_string("alora") == AdapterType.ALORA
        assert AdapterType.from_string("LORA") == AdapterType.LORA
        assert AdapterType.from_string("ALORA") == AdapterType.ALORA

    def test_from_string_invalid(self):
        """Test from_string() with invalid input."""
        with pytest.raises(ValueError, match="Unknown adapter type"):
            AdapterType.from_string("invalid")


class TestAdapterRepoRegistry:
    """Test adapter repository registry."""

    def test_default_initialization(self):
        """Test registry initializes with default IBM repository."""
        registry = AdapterRepoRegistry()
        assert len(registry._repos) == 1
        assert isinstance(registry._repos[0], HuggingFaceAdapterRepo)

    def test_register_repo(self):
        """Test registering additional repositories."""
        registry = AdapterRepoRegistry()
        custom_repo = HuggingFaceAdapterRepo("test-org/test-repo")
        registry.register_repo(custom_repo)
        assert len(registry._repos) == 2
        assert registry._repos[1] == custom_repo

    def test_search_all_types(self):
        """Test searching for all adapter types."""
        registry = AdapterRepoRegistry()
        # This will make real API calls - may want to mock in production
        # For now, just test the interface
        results = registry.search("answerability", "granite-4.0-micro")
        assert isinstance(results, list)
        # Results may be empty if intrinsic doesn't exist for this model

    def test_search_specific_type(self):
        """Test searching for specific adapter type."""
        registry = AdapterRepoRegistry()
        results = registry.search(
            "answerability", "granite-4.0-micro", AdapterType.ALORA
        )
        assert isinstance(results, list)


class TestAdapterResolutionTable:
    """Test adapter resolution table."""

    def test_initialization(self):
        """Test basic initialization."""
        # This will discover adapters - may make API calls
        table = AdapterResolutionTable("granite-4.0-micro")
        assert table.model_id == "granite-4.0-micro"
        assert isinstance(table._implementations, dict)

    def test_resolution_preferences(self):
        """Test that resolution prefers embedded > external, ALORA > LORA."""
        # This is more of an integration test and would need mocking
        # to test thoroughly. Placeholder for now.
        pass

    def test_resolve_nonexistent(self):
        """Test resolving an intrinsic that doesn't exist."""
        table = AdapterResolutionTable("granite-4.0-micro")
        with pytest.raises(ValueError, match="No adapter found"):
            table.resolve("nonexistent_intrinsic")

    def test_mark_loaded(self):
        """Test marking adapter as loaded."""
        table = AdapterResolutionTable("granite-4.0-micro")
        # Add a mock implementation
        table._implementations[("test", AdapterType.LORA)] = AdapterImplementation(
            intrinsic_name="test",
            model_id="granite-4.0-micro",
            technology=AdapterType.LORA,
            implementation_type="external",
            source_repo="test/repo",
            status=AdapterStatus.NOT_LOADED,
        )

        table.mark_loaded("test", AdapterType.LORA)
        impl = table._implementations[("test", AdapterType.LORA)]
        assert impl.status == AdapterStatus.LOADED

    def test_mark_failed(self):
        """Test marking adapter as failed."""
        table = AdapterResolutionTable("granite-4.0-micro")
        # Add a mock implementation
        table._implementations[("test", AdapterType.LORA)] = AdapterImplementation(
            intrinsic_name="test",
            model_id="granite-4.0-micro",
            technology=AdapterType.LORA,
            implementation_type="external",
            source_repo="test/repo",
            status=AdapterStatus.NOT_LOADED,
        )

        table.mark_failed("test", AdapterType.LORA)
        impl = table._implementations[("test", AdapterType.LORA)]
        assert impl.status == AdapterStatus.FAILED


class TestAdapterImplementation:
    """Test AdapterImplementation model."""

    def test_embedded_adapter(self):
        """Test creating embedded adapter implementation."""
        impl = AdapterImplementation(
            intrinsic_name="answerability",
            model_id="granite-with-all-aloras",
            technology=AdapterType.ALORA,
            implementation_type="embedded",
            chat_template_kwargs={"intrinsic_name": "answerability"},
            status=AdapterStatus.LOADED,
            config={"test": "config"},
        )

        assert impl.intrinsic_name == "answerability"
        assert impl.implementation_type == "embedded"
        assert impl.source_repo is None
        assert impl.status == AdapterStatus.LOADED

    def test_external_adapter(self):
        """Test creating external adapter implementation."""
        impl = AdapterImplementation(
            intrinsic_name="answerability",
            model_id="granite-4.0-micro",
            technology=AdapterType.LORA,
            implementation_type="external",
            source_repo="ibm-granite/granite-lib-rag-r1.0",
            status=AdapterStatus.NOT_LOADED,
        )

        assert impl.intrinsic_name == "answerability"
        assert impl.implementation_type == "external"
        assert impl.source_repo == "ibm-granite/granite-lib-rag-r1.0"
        assert impl.chat_template_kwargs is None
        assert impl.status == AdapterStatus.NOT_LOADED
