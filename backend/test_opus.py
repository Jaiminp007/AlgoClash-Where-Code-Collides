#!/usr/bin/env python3
"""
Test script specifically for debugging Claude Opus algorithm generation
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from open_router.algo_gen import generate_algorithm, build_generation_prompt, build_diversity_directives

def test_opus():
    """Test Opus generation with debug output"""

    # Opus model ID
    opus_model = "anthropic/claude-opus-4"
    ticker = "AAPL"

    print("="*80)
    print("TESTING CLAUDE OPUS ALGORITHM GENERATION")
    print("="*80)

    # Build prompt
    print("\n1. Building prompt...")
    base_prompt = build_generation_prompt(ticker, "")
    diversity = build_diversity_directives(opus_model)
    full_prompt = base_prompt + diversity

    print(f"   Prompt length: {len(full_prompt)} characters ({len(full_prompt.split())} words)")
    print(f"   Prompt lines: {len(full_prompt.split(chr(10)))}")

    # Show first part of prompt
    print("\n2. Prompt preview (first 300 chars):")
    print("-" * 80)
    print(full_prompt[:300])
    print("...")
    print("-" * 80)

    # Show diversity directive
    print("\n3. Assigned strategy:")
    print("-" * 80)
    print(diversity)
    print("-" * 80)

    # Attempt generation
    print("\n4. Attempting generation with Opus...")
    print("   (This may take 30-60 seconds)")
    print("-" * 80)

    try:
        code = generate_algorithm(opus_model, full_prompt)

        if code:
            print("\n✅ SUCCESS! Opus generated code.")
            print(f"   Code length: {len(code)} characters")
            print(f"   Code lines: {len(code.split(chr(10)))}")
            print("\n5. Generated code preview (first 500 chars):")
            print("-" * 80)
            print(code[:500])
            print("...")
            print("-" * 80)

            # Save to file
            output_file = Path(__file__).parent / "generate_algo" / "test_opus_output.py"
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(code)
            print(f"\n   Full code saved to: {output_file}")

            # Validate
            from open_router.algo_gen import _validate_generated_code
            is_valid, msg = _validate_generated_code(code, opus_model)

            print("\n6. Validation result:")
            print("-" * 80)
            if is_valid:
                print(f"✅ VALID: {msg}")
            else:
                print(f"❌ INVALID: {msg}")
            print("-" * 80)

        else:
            print("\n❌ FAILED: Opus did not generate code.")
            print("   Check the debug output above to see what Opus returned.")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ ERROR: OPENROUTER_API_KEY environment variable not set!")
        print("   Set it with: export OPENROUTER_API_KEY='your-key-here'")
        sys.exit(1)

    test_opus()
