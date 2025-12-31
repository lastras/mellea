"""Adapter resolution table for tracking available implementations and their states.

This module provides the resolution layer that sits on top of the registry,
tracking which adapters are available for a specific model and their loading states.
"""

from typing import Literal

import pydantic

from .catalog import AdapterStatus, AdapterType, known_intrinsic_names
from .registry import AdapterRepoRegistry, get_default_registry


class AdapterImplementation(pydantic.BaseModel):
    """A specific implementation of an intrinsic for a model.

    Represents either an embedded adapter (built into model checkpoint) or
    an external adapter (to be downloaded from a repository).

    Attributes:
        intrinsic_name: Name of the intrinsic (e.g., "answerability")
        model_id: Model identifier this adapter is for
        technology: Adapter type (LORA or ALORA)
        implementation_type: "embedded" or "external"
        chat_template_kwargs: For embedded adapters, kwargs to pass to chat template
        source_repo: For external adapters, repository ID to download from
        status: Loading state (NOT_LOADED, LOADED, FAILED)
        config: Loaded io.yaml configuration dict
    """

    intrinsic_name: str
    model_id: str
    technology: AdapterType
    implementation_type: Literal["embedded", "external"]

    # For embedded adapters
    chat_template_kwargs: dict | None = None

    # For external adapters
    source_repo: str | None = None

    # State tracking
    status: AdapterStatus = AdapterStatus.NOT_LOADED
    config: dict | None = None


class AdapterResolutionTable:
    """Tracks available adapter implementations for a specific model.

    Discovers both embedded adapters (from adapter_index.json) and external
    adapters (from registry) during initialization. Provides resolution logic
    to select the best available implementation.

    Attributes:
        model_id: Model identifier for this table
        registry: Adapter repository registry for discovering external adapters
        _implementations: Dict mapping (intrinsic_name, technology) to AdapterImplementation
    """

    def __init__(
        self, model_id: str, registry: AdapterRepoRegistry | None = None
    ):
        """Initialize resolution table for a model.

        Discovers all available adapters (embedded and external) during initialization.

        Args:
            model_id: Model identifier (path or HuggingFace ID)
            registry: Optional custom registry. If None, uses global default.

        Example:
            >>> table = AdapterResolutionTable("./granite-with-all-aloras")
            >>> impl = table.resolve("answerability")
            >>> impl.implementation_type
            'embedded'
        """
        self.model_id = model_id
        self.registry = registry or get_default_registry()
        self._implementations: dict[
            tuple[str, AdapterType], AdapterImplementation
        ] = {}

        # Discover everything upfront (no lazy loading)
        self._populate_table()

    def _populate_table(self) -> None:
        """Discover all available adapters.

        Searches in priority order:
        1. Embedded adapters (highest priority)
        2. External adapters from registry
        """
        self._discover_embedded_adapters()
        self._discover_external_adapters()

    def _discover_embedded_adapters(self) -> None:
        """Discover embedded adapters from adapter_index.json.

        Reads adapter_index.json from model directory and creates
        AdapterImplementation entries for each embedded adapter.
        """
        from granite_common.intrinsics.embedded import (
            get_embedded_io_yaml,
            load_adapter_index,
        )
        from granite_common.intrinsics.util import make_config_dict

        adapter_index = load_adapter_index(self.model_id)
        if not adapter_index:
            return  # No embedded adapters

        for adapter_info in adapter_index.get("adapters", []):
            intrinsic_name = adapter_info["intrinsic_name"]
            technology = AdapterType.from_string(adapter_info["technology"])

            # Load io.yaml configuration
            io_yaml_path = get_embedded_io_yaml(self.model_id, intrinsic_name)
            if not io_yaml_path:
                # Embedded adapter missing io.yaml - skip
                continue

            try:
                config_dict = make_config_dict(config_file=io_yaml_path)
            except Exception:
                # Failed to load config - skip this adapter
                continue

            # Build chat template kwargs from config
            chat_template_kwargs = self._build_chat_template_kwargs(
                intrinsic_name, technology, config_dict
            )

            # Create implementation entry
            impl = AdapterImplementation(
                intrinsic_name=intrinsic_name,
                model_id=self.model_id,
                technology=technology,
                implementation_type="embedded",
                chat_template_kwargs=chat_template_kwargs,
                status=AdapterStatus.LOADED,  # Already in model
                config=config_dict,
            )

            self._implementations[(intrinsic_name, technology)] = impl

    def _build_chat_template_kwargs(
        self, intrinsic_name: str, technology: AdapterType, config_dict: dict
    ) -> dict:
        """Build chat template kwargs from adapter config.

        Args:
            intrinsic_name: Name of the intrinsic
            technology: Adapter type
            config_dict: Loaded io.yaml configuration

        Returns:
            Dict of kwargs to pass to chat template

        Note:
            Currently returns minimal kwargs. Can be extended to include
            additional config-based parameters.
        """
        return {"intrinsic_name": intrinsic_name}

    def _discover_external_adapters(self) -> None:
        """Discover external adapters from registry.

        Searches registry for all known intrinsics and populates table
        with AdapterImplementation entries for available external adapters.
        """
        for intrinsic_name in known_intrinsic_names():
            # Search all adapter types
            sources = self.registry.search(intrinsic_name, self.model_id)

            for source in sources:
                key = (intrinsic_name, source.technology)

                # Don't overwrite embedded adapters
                if key in self._implementations:
                    continue

                impl = AdapterImplementation(
                    intrinsic_name=intrinsic_name,
                    model_id=self.model_id,
                    technology=source.technology,
                    implementation_type="external",
                    source_repo=source.repo_id,
                    status=AdapterStatus.NOT_LOADED,
                )

                self._implementations[key] = impl

    def resolve(self, intrinsic_name: str) -> AdapterImplementation:
        """Get best available implementation for an intrinsic.

        Resolution preferences (in order):
        1. Embedded over external
        2. ALORA over LORA

        Args:
            intrinsic_name: Name of intrinsic to resolve

        Returns:
            AdapterImplementation with highest priority

        Raises:
            ValueError: No implementation available for intrinsic

        Example:
            >>> table = AdapterResolutionTable("granite-4.0-micro")
            >>> impl = table.resolve("answerability")
            >>> impl.technology  # Prefers ALORA if both available
            <AdapterType.ALORA: 'alora'>
        """
        candidates = [
            impl
            for (name, tech), impl in self._implementations.items()
            if name == intrinsic_name
        ]

        if not candidates:
            raise ValueError(
                f"No adapter found for intrinsic '{intrinsic_name}' "
                f"on model '{self.model_id}'"
            )

        # Sort by preference: embedded > external, ALORA > LORA
        def sort_key(impl: AdapterImplementation) -> tuple[int, int]:
            return (
                0 if impl.implementation_type == "embedded" else 1,
                0 if impl.technology == AdapterType.ALORA else 1,
            )

        candidates.sort(key=sort_key)
        return candidates[0]

    def mark_loaded(
        self, intrinsic_name: str, technology: AdapterType
    ) -> None:
        """Mark an external adapter as loaded.

        Updates the status of an adapter implementation after it has been
        successfully loaded.

        Args:
            intrinsic_name: Name of the intrinsic
            technology: Adapter type that was loaded

        Example:
            >>> table.mark_loaded("answerability", AdapterType.LORA)
            >>> impl = table._implementations[("answerability", AdapterType.LORA)]
            >>> impl.status
            <AdapterStatus.LOADED: 'loaded'>
        """
        key = (intrinsic_name, technology)
        if key in self._implementations:
            self._implementations[key].status = AdapterStatus.LOADED

    def mark_failed(
        self, intrinsic_name: str, technology: AdapterType, error: str | None = None
    ) -> None:
        """Mark an external adapter as failed to load.

        Args:
            intrinsic_name: Name of the intrinsic
            technology: Adapter type that failed
            error: Optional error message (currently not stored)
        """
        key = (intrinsic_name, technology)
        if key in self._implementations:
            self._implementations[key].status = AdapterStatus.FAILED
