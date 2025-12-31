# Adapter Resolution System

## Executive Summary

This document proposes a new architecture for discovering and resolving intrinsic adapters in Mellea. The system addresses three key limitations in the current implementation:

1. **No support for embedded adapters** - Cannot detect adapters built into model checkpoints (e.g., Granite Switch models)
2. **Hardcoded adapter sources** - Single HuggingFace repository, no extensibility
3. **Eager downloads** - Downloads adapters even when embedded versions are available

The proposed solution introduces a two-tier architecture:
- **Tier 1: AdapterRepoRegistry** - Extensible discovery layer for finding adapters across multiple sources
- **Tier 2: AdapterResolutionTable** - Resolution layer that tracks available implementations and their states

## Current Architecture

### How Intrinsics Work Today

```
User calls: rag.check_answerability(question, documents, context, backend)
    ↓
_call_intrinsic() creates GraniteCommonAdapter manually
    ↓
Hardcoded: adapter_type=AdapterType.LORA
    ↓
backend.add_adapter(adapter) if not already present
    ↓
Backend calls obtain_lora() → downloads from HuggingFace
    ↓
IntrinsicsRewriter transforms chat messages
    ↓
Backend sends to model
```

### Current Limitations

1. **Single source**: Only `ibm-granite/granite-lib-rag-r1.0` repository
2. **Hardcoded types**: Each intrinsic function hardcodes LORA, breaks with ALORA-only models
3. **No embedded detection**: Downloads even when adapter is already in model checkpoint
4. **Manual adapter creation**: Every intrinsic function must create `GraniteCommonAdapter`
5. **No preference logic**: Cannot prefer embedded over external, or ALORA over LORA

### Example: Current Code

```python
# In mellea/stdlib/intrinsics/rag.py
def _call_intrinsic(intrinsic_name, context, backend, kwargs=None):
    # ❌ Hardcoded adapter type
    adapter = GraniteCommonAdapter(
        intrinsic_name,
        adapter_type=AdapterType.LORA,  # What if model only has ALORA?
        base_model_name=backend.base_model_name
    )

    # ❌ Manual check and registration
    if adapter.qualified_name not in backend.list_adapters():
        backend.add_adapter(adapter)  # Triggers download even if embedded
```

## Proposed Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Mellea Backend Init                       │
│                                                              │
│  1. Create AdapterRepoRegistry                              │
│     - HuggingFaceAdapterRepo (ibm-granite/granite-lib-rag) │
│     - [Future: LocalAdapterRepo, CustomRepos, ...]         │
│                                                              │
│  2. Create AdapterResolutionTable                           │
│     - Discover embedded (from adapter_index.json)          │
│     - Discover external (search registry)                  │
│     - Build table of available implementations             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Intrinsic Invocation                      │
│                                                              │
│  User: rag.check_answerability(...)                        │
│    ↓                                                         │
│  Backend: resolve("answerability") → AdapterImplementation │
│    ↓                                                         │
│  If embedded: use chat_template_kwargs                     │
│  If external: load adapter (if not loaded) → use transform │
└─────────────────────────────────────────────────────────────┘
```

### Two-Tier Architecture

#### Tier 1: AdapterRepoRegistry (Discovery)

**Purpose**: Maintain searchable repositories in priority order (like Python's `sys.path`)

**Key Classes**:

```python
@dataclass
class AdapterSource:
    """Minimal information to locate an external adapter."""
    intrinsic_name: str
    model_id: str
    technology: AdapterType  # LORA or ALORA
    repo_id: str  # e.g., "ibm-granite/rag-intrinsics-lib"
    # No path needed - obtain_lora() computes it

class AdapterRepo(abc.ABC):
    """Abstract base for adapter repositories."""

    @abc.abstractmethod
    def find_adapter(
        self,
        intrinsic_name: str,
        model_id: str,
        technology: AdapterType
    ) -> AdapterSource | None:
        """Find adapter in this repository, or None if not available."""
        pass

class HuggingFaceAdapterRepo(AdapterRepo):
    """Repository on HuggingFace Hub."""

    def __init__(self, repo_id: str):
        self.repo_id = repo_id

    def find_adapter(self, intrinsic_name, model_id, technology):
        # Use granite-common's validate_lora_exists() to check without downloading
        from granite_common.intrinsics import validate_lora_exists

        exists = validate_lora_exists(
            intrinsic_name=intrinsic_name,
            target_model_name=model_id,
            repo_id=self.repo_id,
            alora=(technology == AdapterType.ALORA)
        )

        if exists:
            return AdapterSource(
                intrinsic_name=intrinsic_name,
                model_id=model_id,
                technology=technology,
                repo_id=self.repo_id
            )
        return None

class AdapterRepoRegistry:
    """Registry of adapter repositories in priority order."""

    def __init__(self):
        self._repos: list[AdapterRepo] = []
        # Default: IBM Granite intrinsics library
        self.register_repo(
            HuggingFaceAdapterRepo("ibm-granite/granite-lib-rag-r1.0")
        )

    def register_repo(self, repo: AdapterRepo):
        """Add repo to registry (earlier = higher priority)."""
        self._repos.append(repo)

    def search(
        self,
        intrinsic_name: str,
        model_id: str,
        technology: AdapterType | None = None
    ) -> list[AdapterSource]:
        """Search all repos for matching adapters."""
        results = []

        # If technology not specified, search all types
        technologies = [technology] if technology else AdapterType.all_types()

        for tech in technologies:
            for repo in self._repos:
                source = repo.find_adapter(intrinsic_name, model_id, tech)
                if source:
                    results.append(source)

        return results
```

**Extensibility Examples**:

```python
# Add custom private repository
registry.register_repo(HuggingFaceAdapterRepo("myorg/private-adapters"))

# Add local adapter directory (future)
registry.register_repo(LocalAdapterRepo("/path/to/adapters"))

# Search order is insertion order (like sys.path)
```

#### Tier 2: AdapterResolutionTable (Resolution & State)

**Purpose**: Track available implementations and their loading states for this backend

**Key Classes**:

```python
class AdapterStatus(enum.Enum):
    """Lifecycle state of an adapter."""
    NOT_LOADED = "not_loaded"  # Discovered but not loaded
    LOADED = "loaded"          # Ready to use
    FAILED = "failed"          # Load attempt failed

class AdapterImplementation(pydantic.BaseModel):
    """A specific implementation of an intrinsic."""
    intrinsic_name: str
    model_id: str
    technology: AdapterType  # LORA or ALORA
    implementation_type: Literal["embedded", "external"]

    # For embedded adapters
    chat_template_kwargs: dict | None = None

    # For external adapters
    source_repo: str | None = None

    # State tracking
    status: AdapterStatus = AdapterStatus.NOT_LOADED
    config: dict | None = None  # io.yaml configuration

class AdapterResolutionTable:
    """Tracks available adapter implementations for a backend."""

    def __init__(self, model_id: str, registry: AdapterRepoRegistry | None = None):
        self.model_id = model_id
        self.registry = registry or get_default_registry()
        self._implementations: dict[tuple[str, AdapterType], AdapterImplementation] = {}

        # Discover everything upfront (no lazy loading)
        self._populate_table()

    def _populate_table(self):
        """Discover all available adapters."""
        self._discover_embedded_adapters()  # First (highest priority)
        self._discover_external_adapters()  # Second

    def _discover_embedded_adapters(self):
        """Read adapter_index.json from model directory."""
        from granite_common.intrinsics.embedded import (
            load_adapter_index,
            get_embedded_io_yaml
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
                continue

            config_dict = make_config_dict(config_file=io_yaml_path)

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
                config=config_dict
            )

            self._implementations[(intrinsic_name, technology)] = impl

    def _discover_external_adapters(self):
        """Search registry for available external adapters."""
        from granite_common.intrinsics.catalog import known_intrinsic_names

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
                    status=AdapterStatus.NOT_LOADED
                )

                self._implementations[key] = impl

    def resolve(self, intrinsic_name: str) -> AdapterImplementation:
        """Get best available implementation for an intrinsic.

        Resolution preferences (in order):
        1. Embedded over external
        2. ALORA over LORA

        Raises:
            ValueError: No implementation available for intrinsic
        """
        candidates = [
            impl for (name, tech), impl in self._implementations.items()
            if name == intrinsic_name
        ]

        if not candidates:
            raise ValueError(
                f"No adapter found for intrinsic '{intrinsic_name}' "
                f"on model '{self.model_id}'"
            )

        # Sort by preference: embedded > external, ALORA > LORA
        def sort_key(impl: AdapterImplementation):
            return (
                0 if impl.implementation_type == "embedded" else 1,
                0 if impl.technology == AdapterType.ALORA else 1,
            )

        candidates.sort(key=sort_key)
        return candidates[0]

    def mark_loaded(self, intrinsic_name: str, technology: AdapterType):
        """Mark external adapter as loaded."""
        key = (intrinsic_name, technology)
        if key in self._implementations:
            self._implementations[key].status = AdapterStatus.LOADED
```

### Integration with Backends

#### OpenAI Backend (vLLM use case)

```python
class OpenAIBackend:
    def __init__(self, model_id, formatter, base_url, api_key, **kwargs):
        # ... existing initialization ...

        # Create resolution table (discovers all adapters)
        self._adapter_table = AdapterResolutionTable(model_id)

    def generate_from_messages(self, messages, action, **kwargs):
        # Resolve which implementation to use
        impl = self._adapter_table.resolve(intrinsic_name=action.intrinsic_name)

        if impl.implementation_type == "embedded":
            # Use chat template with control token
            rewritten = {
                "messages": messages,
                "extra_body": {
                    "chat_template_kwargs": impl.chat_template_kwargs
                }
            }
        elif impl.implementation_type == "external":
            # Load adapter if needed
            if impl.status != AdapterStatus.LOADED:
                self._load_external_adapter(impl)
                self._adapter_table.mark_loaded(
                    impl.intrinsic_name,
                    impl.technology
                )

            # Use IntrinsicsRewriter transformation
            rewriter = IntrinsicsRewriter(impl.config)
            rewritten = rewriter.rewrite(messages)

        return await self._client.chat.completions.create(**rewritten)

    def _load_external_adapter(self, impl: AdapterImplementation):
        """Download and configure external adapter."""
        from granite_common.intrinsics import obtain_lora, obtain_io_yaml
        from granite_common.intrinsics.util import make_config_dict

        # obtain_lora() handles path resolution and downloading
        adapter_dir = obtain_lora(
            intrinsic_name=impl.intrinsic_name,
            target_model_name=impl.model_id,
            repo_id=impl.source_repo,
            alora=(impl.technology == AdapterType.ALORA)
        )

        # Load configuration
        io_yaml_path = adapter_dir / "io.yaml"
        impl.config = make_config_dict(config_file=io_yaml_path)
```

#### Simplified Intrinsic Functions

```python
# Before: Manual adapter creation with hardcoded type
def check_answerability(question, documents, context, backend):
    adapter = GraniteCommonAdapter(
        "answerability",
        adapter_type=AdapterType.LORA,  # ❌ Hardcoded
        base_model_name=backend.base_model_name
    )
    if adapter.qualified_name not in backend.list_adapters():
        backend.add_adapter(adapter)
    # ... rest of function

# After: Resolution table handles everything
def check_answerability(question, documents, context, backend):
    # Backend's resolution table already has the adapter registered!
    # Just call _call_intrinsic, which uses backend's resolution
    return _call_intrinsic("answerability", context, backend)
```

## granite-common Changes

### New Function: `validate_lora_exists()`

Add to `granite_common/intrinsics/util.py`:

```python
def validate_lora_exists(
    intrinsic_name: str,
    target_model_name: str,
    repo_id: str,
    /,
    alora: bool = False,
) -> bool:
    """
    Check if a LoRA/aLoRA adapter exists in a HuggingFace repository without downloading.

    Uses the same path resolution logic as obtain_lora() but only validates existence.

    :param intrinsic_name: Short name of the intrinsic model
    :param target_model_name: Name of the base model for the adapter
    :param repo_id: HuggingFace Hub repository ID
    :param alora: If True, check for aLoRA; otherwise check for LoRA

    :returns: True if the adapter exists in the repository, False otherwise

    Example:
        >>> validate_lora_exists("answerability", "granite-4.0-micro",
        ...                      "ibm-granite/granite-lib-rag-r1.0", alora=True)
        True
    """
    import huggingface_hub

    # Use same normalization and path logic as obtain_lora()
    target_model_name = BASE_MODEL_TO_CANONICAL_NAME.get(
        target_model_name, target_model_name
    )
    lora_str = "alora" if alora else "lora"

    if repo_id in OLD_LAYOUT_REPOS:
        lora_subdir_name = f"{intrinsic_name}/{lora_str}/{target_model_name}"
    else:
        lora_subdir_name = f"{intrinsic_name}/{target_model_name}/{lora_str}"

    # Check if io.yaml exists in that path
    io_yaml_path = f"{lora_subdir_name}/io.yaml"

    try:
        return huggingface_hub.file_exists(repo_id=repo_id, filename=io_yaml_path)
    except Exception:
        # Repo doesn't exist or other error
        return False
```

**Rationale**: Mellea needs to discover available adapters without triggering downloads. This function validates adapter existence using the HuggingFace Hub API, avoiding unnecessary network I/O.

## Implementation Phases

### Phase 1: granite-common (1 commit)

**Commit**: `Add validate_lora_exists() for adapter discovery`

**Changes**:
- Add `validate_lora_exists()` to `util.py`
- Export from `__init__.py`
- Add docstring with examples

**Files**:
- `granite_common/intrinsics/util.py`
- `granite_common/intrinsics/__init__.py`

### Phase 2: mellea - Foundation (3 commits)

**Commit 2.1**: `Add AdapterType.all_types() and from_string() helpers`

**Changes**:
- Add class methods to `AdapterType` enum
- Remove hardcoded `[AdapterType.LORA, AdapterType.ALORA]` lists

**Files**:
- `mellea/backends/adapters/catalog.py`

**Commit 2.2**: `Add AdapterStatus enum`

**Changes**:
- Define `AdapterStatus` enum (NOT_LOADED, LOADED, FAILED)

**Files**:
- `mellea/backends/adapters/catalog.py`

**Commit 2.3**: `Implement AdapterRepoRegistry`

**Changes**:
- Create `registry.py` with:
  - `AdapterSource` dataclass
  - `AdapterRepo` abstract base
  - `HuggingFaceAdapterRepo` implementation
  - `AdapterRepoRegistry` with search

**Files**:
- `mellea/backends/adapters/registry.py` (new)

### Phase 3: mellea - Resolution Table (1 commit)

**Commit 3.1**: `Implement AdapterResolutionTable`

**Changes**:
- Create `resolution.py` with:
  - `AdapterImplementation` Pydantic model
  - `AdapterResolutionTable` class
  - Embedded discovery via `adapter_index.json`
  - External discovery via registry
  - `resolve()` with preference logic (embedded > external, ALORA > LORA)

**Files**:
- `mellea/backends/adapters/resolution.py` (new)

### Phase 4: mellea - Backend Integration (2 commits)

**Commit 4.1**: `Integrate AdapterResolutionTable with OpenAI backend`

**Changes**:
- Create resolution table in `__init__()`
- Update `generate_from_messages()` to use `resolve()`
- Add `_load_external_adapter()` method

**Files**:
- `mellea/backends/openai.py`

**Commit 4.2**: `Remove legacy get_adapter_for_intrinsic()`

**Changes**:
- Delete `get_adapter_for_intrinsic()` function
- Update any remaining call sites

**Files**:
- `mellea/backends/adapters/adapter.py`

### Phase 5: mellea - Intrinsics Update (2 commits)

**Commit 5.1**: `Update _call_intrinsic to use backend resolution`

**Changes**:
- Remove manual `GraniteCommonAdapter` creation
- Remove hardcoded adapter types
- Rely on backend's resolution table

**Files**:
- `mellea/stdlib/intrinsics/rag.py`

**Commit 5.2**: `Remove hardcoded adapter types from RAG functions`

**Changes**:
- Update docstrings to reflect automatic resolution
- Remove any remaining type hardcoding

**Files**:
- `mellea/stdlib/intrinsics/rag.py`

### Phase 6: Testing (3 commits)

**Commit 6.1**: `Add AdapterRepoRegistry tests`
**Commit 6.2**: `Add AdapterResolutionTable tests`
**Commit 6.3**: `Update integration tests for automatic discovery`

### Phase 7: Documentation (1 commit)

**Commit 7.1**: `Add adapter resolution architecture documentation`

**Files**:
- `mellea/docs/ADAPTER_RESOLUTION.md` (this document)

## Migration Guide

### For Mellea Users

**Before** (manual adapter management):
```python
from mellea.backends.openai import OpenAIBackend
from mellea.backends.adapters.adapter import GraniteCommonAdapter
from mellea.backends.adapters.catalog import AdapterType
import os

backend = OpenAIBackend(model_id="granite-switch-model", ...)

# Manual: construct io.yaml path
io_yaml = os.path.join(model_id, "io_configs", "answerability", "io.yaml")

# Manual: create adapter with specific type
backend.add_adapter(
    GraniteCommonAdapter(
        intrinsic_name="answerability",
        adapter_type=AdapterType.ALORA,
        config_file=io_yaml
    )
)
```

**After** (automatic discovery):
```python
from mellea.backends.openai import OpenAIBackend

# Automatic: discovers embedded adapters from adapter_index.json
backend = OpenAIBackend(model_id="granite-switch-model", ...)

# That's it! Adapters auto-discovered during init
# Use intrinsics normally - backend handles resolution
```

### For Custom Repositories

```python
from mellea.backends.adapters.registry import (
    get_default_registry,
    HuggingFaceAdapterRepo
)

# Add custom repo to default registry
registry = get_default_registry()
registry.register_repo(HuggingFaceAdapterRepo("myorg/custom-adapters"))

# All backends created after this will search custom repo
backend = OpenAIBackend(...)
```

### For Testing

**Testing with embedded adapters**:
```python
# Verify automatic discovery
backend = OpenAIBackend(model_id="./granite-with-adapters")
impl = backend._adapter_table.resolve("answerability")
assert impl.implementation_type == "embedded"
assert impl.status == AdapterStatus.LOADED
```

**Testing resolution preferences**:
```python
# Table has both LORA and ALORA available
impl = backend._adapter_table.resolve("query_rewrite")
# Automatically prefers ALORA
assert impl.technology == AdapterType.ALORA
```

## Benefits

### 1. Automatic Embedded Adapter Detection

**Before**: Manual io.yaml path construction, manual type specification
**After**: Reads `adapter_index.json`, automatically configures

### 2. No Unnecessary Downloads

**Before**: Downloads from HuggingFace even if adapter is embedded
**After**: Uses embedded version when available, only downloads if needed

### 3. Extensible Repository System

**Before**: Single hardcoded HuggingFace repo
**After**: Register multiple repos, custom sources, local directories

### 4. Intelligent Resolution

**Before**: Hardcoded adapter types per intrinsic
**After**: Automatic preference logic (embedded > external, ALORA > LORA)

### 5. Cleaner Intrinsic Functions

**Before**: 10+ lines of adapter management in every intrinsic
**After**: Single line - backend handles everything

### 6. Better Error Messages

**Before**: Generic "adapter not found"
**After**: "No LORA adapter for 'answerability' on 'granite-4.0-micro'. Available: ALORA (embedded)"

## Trade-offs & Alternatives Considered

### Alternative 1: Lazy Discovery

**Considered**: Discover adapters on-demand when intrinsic is called
**Rejected**:
- Adds complexity (caching, race conditions)
- Slower first call per intrinsic
- Harder to validate setup at backend creation
- Current approach: discover everything upfront during `__init__()`

### Alternative 2: Numeric Priorities

**Considered**: Assign numeric priorities to repos and implementations
**Rejected**:
- Harder to reason about (what's priority 5 vs 7?)
- Python convention uses insertion order (sys.path)
- Current approach: insertion order for repos, simple preference rules for resolution

### Alternative 3: Complex Resolution Logic

**Considered**: Include loaded status, preferred technology, etc. in resolution
**Rejected**:
- Overengineered for current needs
- Two simple rules are sufficient: embedded > external, ALORA > LORA
- Current approach: Simple, predictable preferences

### Alternative 4: Store Paths in AdapterSource

**Considered**: Pre-compute relative paths during discovery
**Rejected**:
- `obtain_lora()` already computes paths correctly
- Duplication of path logic
- Harder to maintain consistency
- Current approach: delegate path computation to `obtain_lora()`

### Alternative 5: Keep `is_embedded` Property

**Considered**: Maintain `GraniteCommonAdapter.is_embedded` alongside new system
**Rejected**:
- Redundant with `AdapterImplementation.implementation_type`
- Confusing to have two ways to check
- Current approach: single source of truth in resolution table

## Future Extensions

### 1. Local Adapter Directories

```python
class LocalAdapterRepo(AdapterRepo):
    """Repository on local filesystem."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def find_adapter(self, intrinsic_name, model_id, technology):
        # Check local directory structure
        adapter_path = self.base_path / intrinsic_name / model_id / technology.value
        if (adapter_path / "io.yaml").exists():
            return AdapterSource(...)
        return None

# Usage
registry.register_repo(LocalAdapterRepo("/opt/adapters"))
```

### 2. Adapter Caching Strategies

```python
class CachedHuggingFaceRepo(HuggingFaceAdapterRepo):
    """Cache validation results to avoid repeated API calls."""

    def __init__(self, repo_id, cache_ttl=3600):
        super().__init__(repo_id)
        self._cache = {}
        self._cache_ttl = cache_ttl
```

### 3. Private Repository Authentication

```python
# Support authenticated repos
registry.register_repo(
    HuggingFaceAdapterRepo(
        repo_id="myorg/private-adapters",
        token=os.environ["HF_TOKEN"]
    )
)
```

### 4. Adapter Versioning

```python
@dataclass
class AdapterSource:
    intrinsic_name: str
    model_id: str
    technology: AdapterType
    repo_id: str
    version: str | None = None  # e.g., "v1.2.0"
```

## Testing Strategy

### Unit Tests

- `test_adapter_registry.py`: Repository search, priority ordering
- `test_adapter_resolution.py`: Resolution logic, preference rules, state tracking
- `test_validate_lora_exists.py`: granite-common validation function

### Integration Tests

- `test_embedded_detection.py`: Automatic discovery from `adapter_index.json`
- `test_external_loading.py`: Download and load external adapters
- `test_mixed_adapters.py`: Some embedded, some external
- `test_resolution_preferences.py`: Verify preference ordering

### End-to-End Tests

- `test_granite_switch_vllm.py`: Full workflow with Granite Switch + vLLM
- `test_rag_intrinsics.py`: All RAG intrinsics with new system

## Backwards Compatibility

### Breaking Changes

1. **`get_adapter_for_intrinsic()` removed**: Replace with `_adapter_table.resolve()`
2. **Manual `add_adapter()` no longer needed**: Automatic discovery
3. **`GraniteCommonAdapter.is_embedded` removed**: Use `AdapterImplementation.implementation_type`

### Migration Period

For gradual migration, we could:
1. Deprecate old functions with warnings (not recommended for fork)
2. Support both paths temporarily (adds complexity)
3. Clean break in new branch (recommended)

**Recommendation**: Since this is being developed in a fork/branch, implement clean break. Maintainers can decide migration strategy when merging.

## Open Questions for Maintainers

1. **Registry initialization**: Should default registry be global singleton or per-backend?
   - Current proposal: Global default via `get_default_registry()`
   - Alternative: Each backend gets isolated registry

2. **Failure handling**: What should happen if embedded io.yaml is missing?
   - Current proposal: Skip that adapter, log warning
   - Alternative: Fail backend initialization

3. **Multiple models per backend**: Should AdapterResolutionTable support multiple model_ids?
   - Current proposal: One model per backend/table
   - Use case: Might want to share adapters across similar models

4. **Intrinsics catalog sync**: How to keep catalog.py in sync with actual repo contents?
   - Current proposal: Manual updates to catalog
   - Alternative: Auto-discover intrinsics from repos

## References

- Granite Switch Architecture: `granite-switch/docs/SWITCH_WORKFLOW.md`
- Current Mellea Adapter System: `mellea/backends/adapters/adapter.py`
- Embedded Adapter Detection: `granite-common/intrinsics/embedded.py`
- Python sys.path: https://docs.python.org/3/library/sys.html#sys.path

## Authors & Acknowledgments

Design discussions with Mellea and granite-common maintainers.
Implementation in fork: [repository links TBD]
