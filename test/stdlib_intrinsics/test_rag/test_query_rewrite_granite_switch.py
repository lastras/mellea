#!/usr/bin/env python3
"""
Test query_rewrite intrinsic with Granite Switch model (embedded adapters).

This test validates the query rewrite intrinsic using embedded adapters:
- Automatic discovery of embedded query_rewrite adapter
- No manual adapter setup required
- Converts ambiguous follow-up questions into standalone questions
- Resolves coreferences and adds missing context

Expected behavior:
1. OpenAI backend initialization discovers embedded adapters
2. Resolution table prefers embedded adapter (LORA for query_rewrite)
3. Intrinsic call succeeds without manual adapter setup
4. No external adapter downloads occur
5. Ambiguous question rewritten with full context
"""

import os
import pytest

# Skip if model not available
# Set GRANITE_SWITCH_MODEL environment variable to the path of your Granite Switch model
# Example: export GRANITE_SWITCH_MODEL=/path/to/granite-with-all-aloras
GRANITE_SWITCH_MODEL = os.environ.get("GRANITE_SWITCH_MODEL")
skip_if_no_model = pytest.mark.skipif(
    GRANITE_SWITCH_MODEL is None or not os.path.exists(GRANITE_SWITCH_MODEL),
    reason="GRANITE_SWITCH_MODEL environment variable not set or model not found",
)


@skip_if_no_model
def test_adapter_resolution_query_rewrite():
    """Test that resolution table discovers query_rewrite adapter."""
    from mellea.backends.adapters.catalog import AdapterStatus, AdapterType
    from mellea.backends.adapters.resolution import AdapterResolutionTable

    # Create resolution table - should auto-discover embedded adapters
    table = AdapterResolutionTable(GRANITE_SWITCH_MODEL)

    # Should discover the embedded query_rewrite adapter
    impl = table.resolve("query_rewrite")

    # Verify it's the embedded version
    assert impl.implementation_type == "embedded", (
        "Should discover embedded adapter (not external)"
    )
    # query_rewrite is LORA (not ALORA like answerability)
    assert impl.technology == AdapterType.LORA, (
        "Granite Switch has LORA version of query_rewrite"
    )
    assert impl.status == AdapterStatus.LOADED, (
        "Embedded adapters should be marked as LOADED (already in model)"
    )
    assert impl.config is not None, "Should have loaded io.yaml config"
    assert impl.chat_template_kwargs is not None, (
        "Should have chat template kwargs for embedded adapter"
    )

    print(f"\n✓ Discovered embedded query_rewrite adapter:")
    print(f"  - Intrinsic: {impl.intrinsic_name}")
    print(f"  - Type: {impl.implementation_type}")
    print(f"  - Technology: {impl.technology.value}")
    print(f"  - Status: {impl.status.value}")


@skip_if_no_model
@pytest.mark.asyncio
async def test_query_rewrite_intrinsic_with_embedded_adapter():
    """
    Test query_rewrite intrinsic using embedded adapter (requires vLLM server).

    This test requires a running vLLM server with the Granite Switch model:

    python -m vllm.entrypoints.openai.api_server \\
        --model <path-to-granite-switch-model> \\
        --port 8000 \\
        --dtype auto

    The test validates:
    1. No manual adapter setup needed
    2. Automatic use of embedded LORA adapter
    3. Successful intrinsic invocation
    4. No external downloads
    5. Proper coreference resolution and context integration

    Test case from model card:
    https://huggingface.co/ibm-granite/granite-lib-rag-r1.0/blob/main/query_rewrite/README.md
    """
    # Check if vLLM server is available
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=2.0)
            if response.status_code != 200:
                pytest.skip("vLLM server not available at http://localhost:8000")
    except (httpx.ConnectError, httpx.TimeoutException):
        pytest.skip("vLLM server not available at http://localhost:8000")

    from mellea.backends.openai import OpenAIBackend
    from mellea.stdlib.base import ChatContext
    from mellea.stdlib.chat import Message
    from mellea.stdlib.intrinsics import rag

    # Create backend - automatic adapter discovery
    backend = OpenAIBackend(
        model_id=GRANITE_SWITCH_MODEL,
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    # Override the HF model ID to match what vLLM expects
    # Extract just the model name from the full path for vLLM
    model_name = os.path.basename(GRANITE_SWITCH_MODEL.rstrip('/'))
    backend._hf_model_id = f"./{model_name}/"

    # Verify embedded adapter was discovered
    impl = backend._resolution_table.resolve("query_rewrite")
    assert impl.implementation_type == "embedded"
    print(f"\n✓ Using embedded adapter: {impl.intrinsic_name} ({impl.technology.value})")

    print("\n" + "=" * 60)
    print("TEST: Query Rewrite with Coreference Resolution")
    print("=" * 60)

    # Build conversation context from model card example
    context = (
        ChatContext()
        .add(Message("assistant", "Welcome to pet questions!"))
        .add(Message("user", "I have two pets, a dog named Rex and a cat named Lucy."))
        .add(
            Message(
                "assistant",
                "Rex spends a lot of time in the backyard and outdoors, "
                "and Luna is always inside.",
            )
        )
        .add(
            Message(
                "user",
                "Sounds good! Rex must love exploring outside, while Lucy "
                "probably enjoys her cozy indoor life.",
            )
        )
    )

    # Ambiguous follow-up question with coreference ("he", "that")
    ambiguous_question = "But is he more likely to get fleas because of that?"

    print(f"\nConversation context:")
    for component in context.as_list():
        if isinstance(component, Message):
            role = component.role
            content = component.content
            # Truncate long messages for display
            if len(content) > 80:
                content = content[:77] + "..."
            print(f"  [{role}]: {content}")

    print(f"\nAmbiguous question: '{ambiguous_question}'")
    print("  - Contains coreference: 'he' → Rex")
    print("  - Contains coreference: 'that' → spending time outdoors")

    # Call query rewrite intrinsic
    rewritten = rag.rewrite_question(ambiguous_question, context, backend)

    print(f"\n✓ Rewritten question: '{rewritten}'")

    # Verify the rewritten question is standalone
    assert isinstance(rewritten, str), "Should return a string"
    assert len(rewritten) > len(ambiguous_question), (
        "Rewritten question should be longer (adds context)"
    )

    # Should resolve "he" to "Rex"
    assert "Rex" in rewritten or "rex" in rewritten.lower(), (
        "Should resolve 'he' coreference to 'Rex'"
    )

    # Should mention outdoor activity or backyard
    outdoor_keywords = ["outdoor", "backyard", "outside", "time outside", "spending time"]
    has_outdoor_context = any(keyword in rewritten.lower() for keyword in outdoor_keywords)
    assert has_outdoor_context, (
        f"Should include outdoor/backyard context. Got: {rewritten}"
    )

    # Should mention fleas (preserve original intent)
    assert "flea" in rewritten.lower(), (
        "Should preserve 'fleas' from original question"
    )

    # Verify no external adapter was loaded (should use embedded)
    loaded_adapters = list(backend._loaded_adapters.keys())
    print(f"\n" + "=" * 60)
    print(f"Loaded adapters: {loaded_adapters}")
    assert len(loaded_adapters) == 0, (
        "Embedded adapters should not require external loading"
    )

    print("\n✓ Successfully used embedded adapter via chat template!")
    print(f"✓ Resolved coreferences and added context")
    print(f"✓ Original: '{ambiguous_question}'")
    print(f"✓ Rewritten: '{rewritten}'")


@skip_if_no_model
@pytest.mark.asyncio
async def test_query_rewrite_no_context_needed():
    """
    Test query_rewrite with a question that's already standalone.

    Some questions don't need rewriting - they're already clear and standalone.
    The model should recognize this and return the question as-is or with minimal changes.
    """
    # Check if vLLM server is available
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=2.0)
            if response.status_code != 200:
                pytest.skip("vLLM server not available at http://localhost:8000")
    except (httpx.ConnectError, httpx.TimeoutException):
        pytest.skip("vLLM server not available at http://localhost:8000")

    from mellea.backends.openai import OpenAIBackend
    from mellea.stdlib.base import ChatContext
    from mellea.stdlib.intrinsics import rag

    backend = OpenAIBackend(
        model_id=GRANITE_SWITCH_MODEL,
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    # Override the HF model ID to match what vLLM expects
    model_name = os.path.basename(GRANITE_SWITCH_MODEL.rstrip('/'))
    backend._hf_model_id = f"./{model_name}/"

    print("\n" + "=" * 60)
    print("TEST: Query Rewrite with Standalone Question")
    print("=" * 60)

    # Empty context - no conversation history
    context = ChatContext()

    # Already standalone question with full context
    standalone_question = "What are the symptoms of canine parvovirus in dogs?"

    print(f"\nStandalone question: '{standalone_question}'")
    print("  - No coreferences")
    print("  - Already has full context")
    print("  - No conversation history to pull from")
    print("  - Expected: Adapter returns EXACTLY the same question")

    # Call query rewrite intrinsic
    rewritten = rag.rewrite_question(standalone_question, context, backend)

    print(f"\n✓ Rewritten question: '{rewritten}'")

    # For standalone questions, the adapter is designed to return exactly the same question
    assert rewritten == standalone_question, (
        f"Standalone questions should be returned unchanged. "
        f"Expected: '{standalone_question}', Got: '{rewritten}'"
    )

    print(f"\n✓ Standalone question returned unchanged (as expected)")
    print(f"✓ Original:  '{standalone_question}'")
    print(f"✓ Rewritten: '{rewritten}'")


if __name__ == "__main__":
    # Allow running individual tests for debugging
    import sys

    if "--discovery-only" in sys.argv:
        print("Running discovery tests only...\n")
        test_adapter_resolution_query_rewrite()
        print("\n✓ Discovery test passed!")
    else:
        # Run with pytest
        pytest.main([__file__, "-v", "-s"])
