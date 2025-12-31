"""Adapter repository registry for discovering adapters across multiple sources.

This module provides an extensible system for discovering adapters from various
repositories, similar to Python's sys.path for module discovery.
"""

import abc
from dataclasses import dataclass

from .catalog import AdapterType


@dataclass
class AdapterSource:
    """Minimal information to locate an external adapter.

    Attributes:
        intrinsic_name: Short name of the intrinsic (e.g., "answerability")
        model_id: Base model ID (e.g., "ibm-granite/granite-4.0-micro")
        technology: Adapter type (LORA or ALORA)
        repo_id: Repository ID (e.g., "ibm-granite/granite-lib-rag-r1.0")

    Note:
        Path computation is delegated to obtain_lora() which handles:
        - Model name normalization via BASE_MODEL_TO_CANONICAL_NAME
        - Repository layout detection (old vs new)
        - Relative path construction
    """

    intrinsic_name: str
    model_id: str
    technology: AdapterType
    repo_id: str


class AdapterRepo(abc.ABC):
    """Abstract base class for adapter repositories.

    Repositories provide a searchable source of adapters. Implementations
    can support various backends (HuggingFace Hub, local filesystem, etc.).
    """

    @abc.abstractmethod
    def find_adapter(
        self, intrinsic_name: str, model_id: str, technology: AdapterType
    ) -> AdapterSource | None:
        """Find a specific adapter in this repository.

        Args:
            intrinsic_name: Name of the intrinsic to find
            model_id: Base model identifier
            technology: Adapter type to search for

        Returns:
            AdapterSource if found, None otherwise

        Note:
            This method should not download or load the adapter, only validate
            its existence in the repository.
        """
        pass


class HuggingFaceAdapterRepo(AdapterRepo):
    """Repository of adapters hosted on HuggingFace Hub.

    Uses the validate_lora_exists() function from granite-common to check
    adapter existence without downloading.

    Attributes:
        repo_id: HuggingFace repository ID (e.g., "ibm-granite/granite-lib-rag-r1.0")
    """

    def __init__(self, repo_id: str):
        """Initialize HuggingFace adapter repository.

        Args:
            repo_id: HuggingFace repository ID containing adapters
        """
        self.repo_id = repo_id

    def find_adapter(
        self, intrinsic_name: str, model_id: str, technology: AdapterType
    ) -> AdapterSource | None:
        """Find adapter in HuggingFace repository.

        Uses granite-common's validate_lora_exists() to check existence
        via HuggingFace Hub API without downloading.

        Args:
            intrinsic_name: Name of the intrinsic
            model_id: Base model identifier
            technology: Adapter type (LORA or ALORA)

        Returns:
            AdapterSource if adapter exists in repo, None otherwise

        Example:
            >>> repo = HuggingFaceAdapterRepo("ibm-granite/granite-lib-rag-r1.0")
            >>> source = repo.find_adapter("answerability", "granite-4.0-micro",
            ...                            AdapterType.ALORA)
            >>> source.repo_id
            'ibm-granite/granite-lib-rag-r1.0'
        """
        from granite_common.intrinsics import validate_lora_exists

        exists = validate_lora_exists(
            intrinsic_name=intrinsic_name,
            target_model_name=model_id,
            repo_id=self.repo_id,
            alora=(technology == AdapterType.ALORA),
        )

        if exists:
            return AdapterSource(
                intrinsic_name=intrinsic_name,
                model_id=model_id,
                technology=technology,
                repo_id=self.repo_id,
            )

        return None


class AdapterRepoRegistry:
    """Registry of adapter repositories in priority order.

    Repositories are searched in insertion order, similar to Python's sys.path.
    Earlier registered repositories have higher priority.

    Attributes:
        _repos: List of registered repositories in priority order
    """

    def __init__(self):
        """Initialize registry with default IBM Granite adapter repository."""
        self._repos: list[AdapterRepo] = []
        # Default: IBM Granite intrinsics library
        self.register_repo(
            HuggingFaceAdapterRepo("ibm-granite/granite-lib-rag-r1.0")
        )

    def register_repo(self, repo: AdapterRepo) -> None:
        """Register an adapter repository.

        Repositories are searched in insertion order. Register higher-priority
        repositories first.

        Args:
            repo: Repository to register

        Example:
            >>> registry = AdapterRepoRegistry()
            >>> # Add custom repo with higher priority
            >>> registry.register_repo(
            ...     HuggingFaceAdapterRepo("myorg/custom-adapters")
            ... )
        """
        self._repos.append(repo)

    def search(
        self,
        intrinsic_name: str,
        model_id: str,
        technology: AdapterType | None = None,
    ) -> list[AdapterSource]:
        """Search all repositories for matching adapters.

        Searches repositories in insertion order. If technology is not specified,
        searches for all adapter types.

        Args:
            intrinsic_name: Name of the intrinsic to find
            model_id: Base model identifier
            technology: Optional specific technology to search for.
                       If None, searches all types via AdapterType.all_types()

        Returns:
            List of AdapterSource entries found across all repositories.
            Empty list if no adapters found.

        Example:
            >>> registry = AdapterRepoRegistry()
            >>> # Find all versions (LORA and ALORA)
            >>> sources = registry.search("answerability", "granite-4.0-micro")
            >>> len(sources)
            2
            >>> # Find specific type
            >>> sources = registry.search("answerability", "granite-4.0-micro",
            ...                          AdapterType.ALORA)
            >>> len(sources)
            1
        """
        results = []

        # If technology not specified, search all types
        technologies = [technology] if technology else AdapterType.all_types()

        for tech in technologies:
            for repo in self._repos:
                source = repo.find_adapter(intrinsic_name, model_id, tech)
                if source:
                    results.append(source)

        return results


# Global default registry instance
_default_registry: AdapterRepoRegistry | None = None


def get_default_registry() -> AdapterRepoRegistry:
    """Get the global default adapter registry.

    Returns a singleton instance shared across the application.
    Useful for registering custom repositories that all backends will use.

    Returns:
        Global AdapterRepoRegistry instance

    Example:
        >>> registry = get_default_registry()
        >>> registry.register_repo(
        ...     HuggingFaceAdapterRepo("myorg/private-adapters")
        ... )
        >>> # All backends created after this will search the custom repo
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = AdapterRepoRegistry()
    return _default_registry
