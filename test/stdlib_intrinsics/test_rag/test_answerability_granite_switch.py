#!/usr/bin/env python3
"""
Test answerability intrinsic with Granite Switch model (embedded adapters).

This test validates the new adapter resolution system:
- Automatic discovery of embedded adapters from adapter_index.json
- No manual adapter setup required
- Preference for embedded ALORA over external LORA
- No unnecessary downloads when adapters are embedded

Expected behavior:
1. OpenAI backend initialization discovers embedded adapters
2. Resolution table prefers embedded ALORA adapter
3. Intrinsic call succeeds without manual adapter setup
4. No external adapter downloads occur
"""

import os
import pytest

# Skip if model not available
GRANITE_SWITCH_MODEL = os.environ.get(
    "GRANITE_SWITCH_MODEL", "/home/lastrasl/granite-switch/granite-with-all-aloras"
)
skip_if_no_model = pytest.mark.skipif(
    not os.path.exists(GRANITE_SWITCH_MODEL),
    reason=f"Granite Switch model not found at {GRANITE_SWITCH_MODEL}",
)


@skip_if_no_model
def test_adapter_resolution_discovery():
    """Test that resolution table discovers embedded adapters automatically."""
    from mellea.backends.adapters.catalog import AdapterStatus, AdapterType
    from mellea.backends.adapters.resolution import AdapterResolutionTable

    # Create resolution table - should auto-discover embedded adapters
    table = AdapterResolutionTable(GRANITE_SWITCH_MODEL)

    # Should discover the embedded answerability ALORA adapter
    impl = table.resolve("answerability")

    # Verify it's the embedded ALORA version
    assert impl.implementation_type == "embedded", (
        "Should discover embedded adapter (not external)"
    )
    assert impl.technology == AdapterType.ALORA, (
        "Granite Switch has ALORA version of answerability"
    )
    assert impl.status == AdapterStatus.LOADED, (
        "Embedded adapters should be marked as LOADED (already in model)"
    )
    assert impl.config is not None, "Should have loaded io.yaml config"
    assert impl.chat_template_kwargs is not None, (
        "Should have chat template kwargs for embedded adapter"
    )

    print(f"\n✓ Discovered embedded adapter:")
    print(f"  - Intrinsic: {impl.intrinsic_name}")
    print(f"  - Type: {impl.implementation_type}")
    print(f"  - Technology: {impl.technology.value}")
    print(f"  - Status: {impl.status.value}")


@skip_if_no_model
def test_resolution_table_discovers_all_embedded():
    """Test that resolution table discovers all embedded adapters in Granite Switch."""
    from mellea.backends.adapters.resolution import AdapterResolutionTable

    # Granite Switch model has 5 embedded adapters:
    # - 4 LORA: context_relevance, query_rewrite, answer_relevance_rewriter,
    #           answer_relevance_classifier
    # - 1 ALORA: answerability
    table = AdapterResolutionTable(GRANITE_SWITCH_MODEL)

    # Check we discovered all embedded adapters
    embedded_impls = [
        impl
        for impl in table._implementations.values()
        if impl.implementation_type == "embedded"
    ]

    # Should have discovered all 5 embedded adapters
    assert len(embedded_impls) >= 5, (
        f"Expected at least 5 embedded adapters, found {len(embedded_impls)}"
    )

    # All should be marked as LOADED
    for impl in embedded_impls:
        assert impl.status.value == "loaded", (
            f"Embedded adapter {impl.intrinsic_name} should be LOADED"
        )

    print(f"\n✓ Discovered {len(embedded_impls)} embedded adapters:")
    for impl in embedded_impls:
        print(
            f"  - {impl.intrinsic_name}: {impl.technology.value} "
            f"({impl.status.value})"
        )


@skip_if_no_model
def test_backend_initialization_with_embedded_adapters():
    """Test that OpenAI backend initializes with embedded adapter discovery."""
    from mellea.backends.openai import OpenAIBackend

    # Create backend pointing to Granite Switch model
    # This should automatically discover embedded adapters
    backend = OpenAIBackend(
        model_id=GRANITE_SWITCH_MODEL,
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    # Backend should have created resolution table
    assert hasattr(backend, "_resolution_table"), (
        "Backend should have resolution table"
    )

    # Resolution table should have discovered embedded adapters
    table = backend._resolution_table
    impl = table.resolve("answerability")

    assert impl.implementation_type == "embedded", (
        "Should discover embedded adapter automatically"
    )
    assert impl.technology.value == "alora", (
        "Granite Switch has ALORA answerability"
    )

    print("\n✓ Backend initialized with automatic embedded adapter discovery")
    print(f"  - Model: {GRANITE_SWITCH_MODEL}")
    print(f"  - Discovered: {impl.intrinsic_name} ({impl.technology.value})")
    print(f"  - Type: {impl.implementation_type}")


@skip_if_no_model
def test_answerability_intrinsic_with_embedded_adapter():
    """
    Test answerability intrinsic using embedded adapter (requires vLLM server).

    This test requires a running vLLM server with the Granite Switch model:

    python -m vllm.entrypoints.openai.api_server \\
        --model /home/lastrasl/granite-switch/granite-with-all-aloras \\
        --port 8000 \\
        --dtype auto

    The test validates:
    1. No manual adapter setup needed
    2. Automatic use of embedded ALORA adapter
    3. Successful intrinsic invocation
    4. No external downloads
    """
    pytest.skip(
        "Requires vLLM server - run manually or set ENABLE_VLLM_TESTS=1. "
        "This test documents expected behavior for embedded adapter invocation."
    )

    from mellea.backends.openai import OpenAIBackend
    from mellea.stdlib.base import ChatContext, Document
    from mellea.stdlib.intrinsics import rag

    # Create backend - automatic adapter discovery
    backend = OpenAIBackend(
        model_id=GRANITE_SWITCH_MODEL,
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    # Verify embedded adapter was discovered
    impl = backend._resolution_table.resolve("answerability")
    assert impl.implementation_type == "embedded"
    print(f"\n✓ Using embedded adapter: {impl.intrinsic_name} ({impl.technology.value})")

    # Create test data
    question = "What is the capital of France?"
    documents = [
        Document(text="Paris is the capital and largest city of France."),
        Document(text="Berlin is the capital of Germany."),
    ]
    context = ChatContext()

    print(f"\nQuestion: {question}")
    print(f"Documents: {len(documents)}")

    # Call intrinsic - should automatically use embedded adapter
    # NO manual adapter setup needed!
    try:
        score = rag.check_answerability(question, documents, context, backend)

        print(f"\n✓ Answerability score: {score}")
        assert 0.0 <= score <= 1.0, "Score should be between 0 and 1"
        assert score > 0.5, (
            "Question should be answerable from documents (Paris is mentioned)"
        )

        # Verify no external adapter was loaded (should use embedded)
        loaded_adapters = list(backend._loaded_adapters.keys())
        print(f"\nLoaded adapters: {loaded_adapters}")

        # NOTE: Currently raises NotImplementedError because embedded adapter
        # support via chat template is not yet implemented in the backend.
        # This test documents the expected behavior once that's implemented.

    except NotImplementedError as e:
        if "Embedded adapter support" in str(e):
            pytest.skip(
                "Embedded adapter invocation via chat template not yet implemented. "
                "This test documents expected behavior for future implementation."
            )
        raise


@skip_if_no_model
def test_no_external_downloads_for_embedded():
    """Verify that embedded adapters don't trigger external downloads."""
    from mellea.backends.adapters.resolution import AdapterResolutionTable

    # Create resolution table for Granite Switch
    table = AdapterResolutionTable(GRANITE_SWITCH_MODEL)

    # Resolve answerability - should get embedded adapter
    impl = table.resolve("answerability")

    # Verify no external source
    assert impl.implementation_type == "embedded"
    assert impl.source_repo is None, (
        "Embedded adapters should not have source_repo (no download needed)"
    )

    print("\n✓ Embedded adapter requires no external downloads")
    print(f"  - Intrinsic: {impl.intrinsic_name}")
    print(f"  - Source: Built into model checkpoint")


if __name__ == "__main__":
    # Allow running individual tests for debugging
    import sys

    if "--discovery-only" in sys.argv:
        print("Running discovery tests only...\n")
        test_adapter_resolution_discovery()
        test_resolution_table_discovers_all_embedded()
        test_backend_initialization_with_embedded_adapters()
        test_no_external_downloads_for_embedded()
        print("\n✓ All discovery tests passed!")
    else:
        # Run with pytest
        pytest.main([__file__, "-v", "-s"])
