"""
Benchmark script to demonstrate concurrent vs sequential algorithm generation.

This script compares the performance of:
1. Sequential generation (old approach)
2. Concurrent generation (new async approach)

Usage:
    python examples/benchmark_async_gen.py
"""

import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from open_router.algo_gen import (
    generate_algorithms_for_agents,
)


def benchmark_concurrent_generation():
    """
    Benchmark concurrent algorithm generation.

    Expected behavior:
    - 6 requests execute in parallel
    - Total time ≈ max(individual request times), not sum(individual request times)
    - With typical API latency of 5-15s per request, concurrent execution
      should complete in 5-15s vs 30-90s for sequential
    """

    # Test models (using free models to avoid API costs)
    test_models = [
        "meta-llama/llama-3.2-1b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "google/gemma-2-9b-it:free",
        "mistralai/mistral-7b-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "qwen/qwen-2-7b-instruct:free",
    ]

    print("=" * 70)
    print("ASYNC ALGORITHM GENERATION BENCHMARK")
    print("=" * 70)
    print()
    print(f"Models to test: {len(test_models)}")
    for i, model in enumerate(test_models, 1):
        print(f"  {i}. {model}")
    print()
    print("This benchmark will:")
    print("  1. Generate algorithms for all 6 models CONCURRENTLY")
    print("  2. Measure total execution time")
    print("  3. Compare against theoretical sequential time")
    print()

    # Progress callback to track individual completions
    completions = []

    def progress_callback(percent, message):
        if "MODEL_OK" in message:
            model_name = message.split("::")[1]
            completions.append((time.time(), model_name))
            print(f"  ✓ Completed: {model_name}")
        elif "FAILED" in message.upper() or "ERROR" in message.upper():
            print(f"  ✗ {message}")

    print("Starting concurrent generation...")
    print("-" * 70)
    start_time = time.time()

    try:
        result = generate_algorithms_for_agents(
            test_models,
            "AAPL",
            progress_callback=progress_callback
        )

        end_time = time.time()
        elapsed = end_time - start_time

        print("-" * 70)
        print()
        print("RESULTS:")
        print("=" * 70)
        print(f"Success: {result}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Successful completions: {len(completions)}/{len(test_models)}")
        print()

        if completions:
            # Calculate time to first and last completion
            first_completion = min(c[0] for c in completions) - start_time
            last_completion = max(c[0] for c in completions) - start_time

            print(f"Time to first completion: {first_completion:.2f}s")
            print(f"Time to last completion: {last_completion:.2f}s")
            print()

            # Estimate sequential time (assume avg 10s per request)
            avg_time_per_request = elapsed / len(completions) if completions else 10
            estimated_sequential = avg_time_per_request * len(test_models)
            speedup = estimated_sequential / elapsed if elapsed > 0 else 0

            print("PERFORMANCE ANALYSIS:")
            print("-" * 70)
            print(f"Avg time per request: {avg_time_per_request:.2f}s")
            print(f"Estimated sequential time: {estimated_sequential:.2f}s")
            print(f"Actual concurrent time: {elapsed:.2f}s")
            print(f"Speedup: {speedup:.2f}x")
            print()

            if speedup > 3:
                print("✅ EXCELLENT: Concurrent execution is working as expected!")
                print(f"   You saved ~{estimated_sequential - elapsed:.0f}s compared to sequential execution.")
            elif speedup > 2:
                print("✓ GOOD: Concurrent execution is providing benefits.")
            else:
                print("⚠ WARNING: Speedup is less than expected.")
                print("   This could be due to API rate limiting or network conditions.")

        print()
        print("=" * 70)

        return result

    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Benchmark failed: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ ERROR: OPENROUTER_API_KEY not found in environment")
        print("Please set your API key in backend/.env file")
        sys.exit(1)

    print(f"API Key found: {api_key[:20]}...")
    print()

    success = benchmark_concurrent_generation()
    sys.exit(0 if success else 1)
