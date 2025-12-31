# Adapter Resolution System - Examples & Test Cases

This document captures test cases and examples from experimental work to guide implementation.

## Test Case 1: Granite Switch with Embedded Adapters

### Setup

Model: `granite-with-all-aloras` (Granite Switch with embedded adapters)
- Has `adapter_index.json` with 5 embedded adapters
- 4 LORA adapters: context_relevance, query_rewrite, answer_relevance_rewriter, answer_relevance_classifier
- 1 ALORA adapter: answerability
- io.yaml files in `io_configs/*/io.yaml`

### Expected Behavior

```python
# Backend initialization auto-discovers embedded adapters
backend = OpenAIBackend(
    model_id="./granite-with-all-aloras",
    formatter=TemplateFormatter(model_id="./granite-with-all-aloras"),
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Verify discovery
impl = backend._adapter_table.resolve("answerability")
assert impl.implementation_type == "embedded"
assert impl.technology == AdapterType.ALORA
assert impl.status == AdapterStatus.LOADED
assert impl.chat_template_kwargs == {"intrinsic_name": "answerability"}
```

### Test File Location
`tests/test_embedded_detection.py`

## Test Case 2: External Adapter Download

### Setup

Model: `ibm-granite/granite-3.3-8b-instruct` (no embedded adapters)
Backend should discover and download from HuggingFace

### Expected Behavior

```python
backend = HuggingFaceBackend(model_id="ibm-granite/granite-3.3-8b-instruct")

# Table discovers external adapters available
impl = backend._adapter_table.resolve("answerability")
assert impl.implementation_type == "external"
assert impl.source_repo == "ibm-granite/granite-lib-rag-r1.0"
assert impl.status == AdapterStatus.NOT_LOADED

# First call triggers download
result = rag.check_answerability(question, documents, context, backend)

# Now marked as loaded
impl = backend._adapter_table.resolve("answerability")
assert impl.status == AdapterStatus.LOADED
```

### Test File Location
`tests/test_external_loading.py`

## Test Case 3: Mixed Embedded and External

### Setup

Two different models: one with embedded adapters (Granite Switch), one without (base Granite)
- `granite-with-all-aloras`: Has 5 embedded adapters
- `ibm-granite/granite-4.0-micro`: Base model with no embedded adapters

### Expected Behavior

```python
# Granite Switch model: has embedded adapters
backend_switch = OpenAIBackend(model_id="./granite-with-all-aloras")

# Embedded adapter - no download
impl1 = backend_switch._adapter_table.resolve("answerability")
assert impl1.implementation_type == "embedded"
assert impl1.technology == AdapterType.ALORA
assert impl1.status == AdapterStatus.LOADED

# Base Granite model: no embedded adapters, needs external download
backend_base = HuggingFaceBackend(model_id="ibm-granite/granite-4.0-micro")

# External adapter - will download from ibm-granite/granite-lib-rag-r1.0
impl2 = backend_base._adapter_table.resolve("answerability")
assert impl2.implementation_type == "external"
assert impl2.source_repo == "ibm-granite/granite-lib-rag-r1.0"
assert impl2.status == AdapterStatus.NOT_LOADED

# After first call, marked as loaded
result = await rag.check_answerability(question, documents, context, backend_base)
impl2 = backend_base._adapter_table.resolve("answerability")
assert impl2.status == AdapterStatus.LOADED
```

### Test File Location
`tests/test_mixed_adapters.py`

## Test Case 4: ALORA vs LORA Preference

### Setup

Repository `ibm-granite/granite-lib-rag-r1.0` has both LORA and ALORA versions for some intrinsics on granite-4.0-micro:
- `answerability`: has both lora and alora
- `context_relevance`: has both lora and alora
- `query_rewrite`: has both lora and alora

### Expected Behavior

```python
# Base Granite model - will discover external adapters
backend = HuggingFaceBackend(model_id="ibm-granite/granite-4.0-micro")

# Table discovers both types for answerability
table = backend._adapter_table

# Can check what's available
implementations = [
    impl for (name, tech), impl in table._implementations.items()
    if name == "answerability"
]
assert len(implementations) == 2  # Both LORA and ALORA available

# Resolution automatically prefers ALORA over LORA
impl = table.resolve("answerability")
assert impl.technology == AdapterType.ALORA
assert impl.implementation_type == "external"
assert impl.source_repo == "ibm-granite/granite-lib-rag-r1.0"

# Same for query_rewrite
impl = table.resolve("query_rewrite")
assert impl.technology == AdapterType.ALORA
```

### Test File Location
`tests/test_resolution_preferences.py`

## Test Case 5: Custom Repository Registration

### Setup

Create a custom repository with subset of adapters as test data:
- Test repo: `test-org/custom-granite-adapters` (or local directory)
- Contains only 2 intrinsics: `answerability` and `context_relevance` for `granite-4.0-micro`
- Both have only LORA versions (no ALORA)

This simulates a custom/private adapter repository scenario.

### Expected Behavior

```python
from mellea.backends.adapters.registry import (
    get_default_registry,
    HuggingFaceAdapterRepo
)

# Register custom repo BEFORE default repo (higher priority)
registry = get_default_registry()
# Clear default repos for test
registry._repos.clear()

# Add custom repo first, then default
registry.register_repo(HuggingFaceAdapterRepo("test-org/custom-granite-adapters"))
registry.register_repo(HuggingFaceAdapterRepo("ibm-granite/granite-lib-rag-r1.0"))

# Create backend with custom registry
backend = HuggingFaceBackend(model_id="ibm-granite/granite-4.0-micro")
table = backend._adapter_table

# For answerability: custom repo has only LORA, default has both LORA and ALORA
# Should find LORA from custom repo (first in registry)
impl = table.resolve("answerability")
assert impl.source_repo == "test-org/custom-granite-adapters"
assert impl.technology == AdapterType.LORA  # Custom repo only has LORA

# For query_rewrite: not in custom repo, falls back to default repo
impl = table.resolve("query_rewrite")
assert impl.source_repo == "ibm-granite/granite-lib-rag-r1.0"
assert impl.technology == AdapterType.ALORA  # Default repo has both, prefers ALORA

# Verify insertion order priority
implementations = [
    impl for (name, tech), impl in table._implementations.items()
    if name == "answerability"
]
# Should have found LORA from custom repo AND ALORA from default repo
# (different technologies don't conflict)
assert len(implementations) == 2
```

### Test File Location
`tests/test_custom_repositories.py`

### Test Setup
To run this test, create a minimal test repository:
```bash
# Option 1: Create local test repo structure
mkdir -p test-adapters/answerability/granite-4.0-micro/lora
cp -r ~/.cache/huggingface/.../answerability/.../lora/* test-adapters/answerability/granite-4.0-micro/lora/

# Option 2: Fork ibm-granite/granite-lib-rag-r1.0 and create subset repo on HuggingFace
```

## Test Case 6: Adapter Not Found Error

### Setup

Request intrinsic that doesn't exist for the model

### Expected Behavior

```python
backend = OpenAIBackend(model_id="some-model")

# No adapter available
with pytest.raises(ValueError, match="No adapter found for intrinsic 'nonexistent'"):
    backend._adapter_table.resolve("nonexistent")

# Error message should be helpful
# "No adapter found for intrinsic 'nonexistent' on model 'some-model'"
```

### Test File Location
`tests/test_error_handling.py`

## Test Case 7: Embedded-Only Discovery (Granite Switch)

### Setup

Granite Switch model with embedded adapters - currently doesn't support loading additional external adapters

### Expected Behavior

```python
# Granite Switch model with 5 embedded adapters
backend = OpenAIBackend(model_id="./granite-with-all-aloras")

# Only embedded adapters discovered
table = backend._adapter_table

# Check all discovered implementations
all_impls = list(table._implementations.values())
assert all(impl.implementation_type == "embedded" for impl in all_impls)
assert len(all_impls) == 5  # Only the 5 embedded adapters

# No external adapters in table (Granite Switch doesn't support external loading yet)
external_impls = [
    impl for impl in all_impls
    if impl.implementation_type == "external"
]
assert len(external_impls) == 0

# Resolution uses embedded adapter
impl = table.resolve("answerability")
assert impl.implementation_type == "embedded"
assert impl.technology == AdapterType.ALORA
assert impl.status == AdapterStatus.LOADED
```

### Test File Location
`tests/test_embedded_only.py`

### Note
Future enhancement: Add support for loading external adapters into Granite Switch models to complement embedded ones.

## Test Case 8: Chat Template Integration

### Setup

Embedded adapter with chat template kwargs

### Expected Behavior

```python
backend = OpenAIBackend(model_id="./granite-switch-model", base_url=vllm_url)

# Call intrinsic function
result = await rag.check_answerability(
    question="Is this answerable?",
    documents=[Document(text="Some context")],
    context=ChatContext(),
    backend=backend
)

# Backend automatically:
# 1. Resolved answerability → embedded ALORA
# 2. Built chat_template_kwargs = {"intrinsic_name": "answerability"}
# 3. Passed to vLLM via extra_body
# 4. vLLM chat template inserted control token
# 5. Model processed with embedded adapter
```

### Test File Location
`tests/test_chat_template_integration.py`

## Example: adapter_index.json Structure

From experimental Granite Switch model:

```json
{
  "model_info": {
    "num_adapters": 5,
    "base_model": "granite-4.0-micro"
  },
  "adapters": [
    {
      "adapter_index": 1,
      "intrinsic_name": "context_relevance",
      "technology": "lora",
      "original_path": "/path/to/cache/...",
      "io_config": "io_configs/context_relevance/io.yaml",
      "control_token": {
        "token": "<context_relevance>",
        "id": 100353
      }
    },
    {
      "adapter_index": 4,
      "intrinsic_name": "answerability",
      "technology": "alora",
      "io_config": "io_configs/answerability/io.yaml",
      "control_token": {
        "token": "<answerability>",
        "id": 100355
      }
    }
  ]
}
```

## Example: Resolution Table State

After initialization with mixed model:

```python
table = AdapterResolutionTable("./granite-with-some-embedded")

# Internal state:
table._implementations = {
    ("answerability", AdapterType.ALORA): AdapterImplementation(
        intrinsic_name="answerability",
        model_id="./granite-with-some-embedded",
        technology=AdapterType.ALORA,
        implementation_type="embedded",
        chat_template_kwargs={"intrinsic_name": "answerability"},
        status=AdapterStatus.LOADED,
        config={...}
    ),
    ("query_rewrite", AdapterType.LORA): AdapterImplementation(
        intrinsic_name="query_rewrite",
        model_id="./granite-with-some-embedded",
        technology=AdapterType.LORA,
        implementation_type="external",
        source_repo="ibm-granite/granite-lib-rag-r1.0",
        status=AdapterStatus.NOT_LOADED,
        config=None
    ),
    ("query_rewrite", AdapterType.ALORA): AdapterImplementation(
        intrinsic_name="query_rewrite",
        model_id="./granite-with-some-embedded",
        technology=AdapterType.ALORA,
        implementation_type="external",
        source_repo="ibm-granite/granite-lib-rag-r1.0",
        status=AdapterStatus.NOT_LOADED,
        config=None
    ),
}

# resolve("answerability") → embedded ALORA (loaded)
# resolve("query_rewrite") → external ALORA (will download on first use)
```

## Key Behaviors to Test

### Registry Tests
- [x] `HuggingFaceAdapterRepo.find_adapter()` returns `AdapterSource` if exists
- [x] Returns `None` if adapter doesn't exist in repo
- [x] Search follows insertion order (earlier repos checked first)
- [x] Can search all technologies or specific one
- [x] Multiple repos can be registered

### Resolution Tests
- [x] Embedded adapters discovered from `adapter_index.json`
- [x] External adapters discovered from registry search
- [x] Embedded adapters not overwritten by external
- [x] Resolution prefers: embedded > external
- [x] Resolution prefers: ALORA > LORA
- [x] `mark_loaded()` updates status correctly
- [x] ValueError raised if no adapter found

### Backend Integration Tests
- [x] Resolution table created during backend init
- [x] Embedded adapters auto-discovered
- [x] External adapters loaded on demand
- [x] Chat template kwargs passed correctly (embedded)
- [x] IntrinsicsRewriter used correctly (external)

### Intrinsics Tests
- [x] `_call_intrinsic()` uses backend resolution
- [x] No manual `GraniteCommonAdapter` creation
- [x] Works with embedded adapters (no download)
- [x] Works with external adapters (downloads)
- [x] All RAG intrinsics work with new system

## Performance Considerations

### Discovery Cost
- **Embedded**: Fast (local file read of `adapter_index.json`)
- **External**: One HF API call per (intrinsic, model, technology) tuple
  - For 7 intrinsics × 1 model × 2 technologies = ~14 API calls
  - Happens once at backend initialization
  - Could add caching in future

### Resolution Cost
- **Embedded**: O(1) dictionary lookup
- **External (not loaded)**: First call downloads adapter
- **External (loaded)**: O(1) dictionary lookup

## Edge Cases

### 1. Model directory doesn't exist
```python
# Should handle gracefully
table = AdapterResolutionTable("/nonexistent/path")
# No embedded adapters discovered, only external
```

### 2. adapter_index.json is malformed
```python
# Should log warning and skip
# Fall back to external adapters only
```

### 3. io.yaml file missing for embedded adapter
```python
# Should skip that adapter entry
# Log warning: "Embedded adapter 'X' missing io.yaml"
```

### 4. Network failure during external validation
```python
# validate_lora_exists() returns False
# Adapter not added to table
# No failure - just not available
```

### 5. Download fails for external adapter
```python
# _load_external_adapter() should:
# 1. Update status to FAILED
# 2. Raise clear error
# 3. User can retry later
```

## Migration Examples

### Before: Manual Adapter Management

```python
# Complex setup with manual path construction
import os
from mellea.backends.openai import OpenAIBackend
from mellea.backends.adapters.adapter import GraniteCommonAdapter
from mellea.backends.adapters.catalog import AdapterType

model_path = "./granite-switch-model"
backend = OpenAIBackend(model_id=model_path, ...)

# Manual detection
io_yaml_path = os.path.join(model_path, "io_configs", "answerability", "io.yaml")

# Manual adapter creation with type specification
adapter = GraniteCommonAdapter(
    intrinsic_name="answerability",
    adapter_type=AdapterType.ALORA,  # Must know type
    config_file=io_yaml_path
)

# Manual registration
backend.add_adapter(adapter)

# Manual check
if adapter.is_embedded:
    print("Adapter is embedded")
```

### After: Automatic Discovery

```python
# Simple - everything automatic
from mellea.backends.openai import OpenAIBackend

backend = OpenAIBackend(model_id="./granite-switch-model", ...)

# All embedded adapters auto-discovered during __init__
# Just use intrinsics normally!
result = await rag.check_answerability(question, documents, context, backend)
```

## Debug / Inspection Helpers

Useful for development and debugging:

```python
# List all discovered adapters
for (name, tech), impl in backend._adapter_table._implementations.items():
    print(f"{name} ({tech.value}): {impl.implementation_type} - {impl.status.value}")

# Check specific intrinsic
impl = backend._adapter_table.resolve("answerability")
print(f"Using {impl.implementation_type} {impl.technology.value}")
print(f"Status: {impl.status.value}")
if impl.implementation_type == "embedded":
    print(f"Chat template kwargs: {impl.chat_template_kwargs}")
else:
    print(f"Source repo: {impl.source_repo}")
```

## Summary

This document provides concrete test cases and examples to validate the adapter resolution system. Each test case should be implemented in the testing phase to ensure the system works correctly across all scenarios:

1. ✅ Embedded adapter detection
2. ✅ External adapter download
3. ✅ Mixed embedded/external
4. ✅ Resolution preferences (embedded > external, ALORA > LORA)
5. ✅ Custom repository registration
6. ✅ Error handling
7. ✅ Chat template integration
8. ✅ Edge cases (missing files, network failures, etc.)
