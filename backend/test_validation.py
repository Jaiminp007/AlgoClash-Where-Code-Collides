#!/usr/bin/env python3
"""
Test script to validate the code validation improvements
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from open_router.algo_gen import _validate_generated_code

def test_validation():
    """Test the validation function with various code samples"""

    print("="*70)
    print("TESTING CODE VALIDATION IMPROVEMENTS")
    print("="*70)

    # Test 1: Empty implementation (like broken Qwen models)
    print("\n1. Testing BROKEN code (empty implementation):")
    print("-" * 70)
    broken_code = "def execute_trade(ticker, cash_balance, shares_held):"
    is_valid, msg = _validate_generated_code(broken_code, "test-broken")
    print(f"Code: {broken_code}")
    print(f"Valid: {is_valid}")
    print(f"Message:\n{msg}")
    assert not is_valid, "Should reject empty implementation"
    print("✅ PASS - Correctly rejected empty code")

    # Test 2: Missing return statements
    print("\n2. Testing code with NO return statements:")
    print("-" * 70)
    no_return = """
def execute_trade(ticker, cash_balance, shares_held):
    import yfinance as yf
    data = yf.download(ticker, end="2007-10-09")
    # Missing return!
"""
    is_valid, msg = _validate_generated_code(no_return, "test-no-return")
    print(f"Valid: {is_valid}")
    print(f"Message:\n{msg}")
    assert not is_valid, "Should reject code without returns"
    print("✅ PASS - Correctly rejected code without returns")

    # Test 3: Missing data cutoff (cheating)
    print("\n3. Testing code with FUTURE DATA LEAKAGE:")
    print("-" * 70)
    cheating_code = """
import yfinance as yf
import numpy as np

def execute_trade(ticker, cash_balance, shares_held):
    # CHEATING: No end= parameter!
    data = yf.download(ticker, start="2023-01-01")
    if len(data) > 0:
        return "BUY"
    return "HOLD"
"""
    is_valid, msg = _validate_generated_code(cheating_code, "test-cheating")
    print(f"Valid: {is_valid}")
    print(f"Message:\n{msg}")
    assert not is_valid, "Should reject code without end= parameter"
    print("✅ PASS - Correctly rejected code with future data leakage")

    # Test 4: Forbidden imports
    print("\n4. Testing code with FORBIDDEN imports:")
    print("-" * 70)
    forbidden_code = """
import yfinance as yf
import sklearn  # FORBIDDEN!

def execute_trade(ticker, cash_balance, shares_held):
    data = yf.download(ticker, end="2007-10-09")
    return "BUY" if len(data) > 0 else "HOLD"
"""
    is_valid, msg = _validate_generated_code(forbidden_code, "test-forbidden")
    print(f"Valid: {is_valid}")
    print(f"Message:\n{msg}")
    assert not is_valid, "Should reject code with forbidden imports"
    print("✅ PASS - Correctly rejected code with forbidden imports")

    # Test 5: Valid code (should pass)
    print("\n5. Testing VALID code:")
    print("-" * 70)
    valid_code = """
import yfinance as yf
import numpy as np

_cache = None

def execute_trade(ticker, cash_balance, shares_held):
    global _cache

    try:
        if _cache is None:
            _cache = yf.download(ticker, start="2023-01-01", end="2007-10-09", progress=False)

        if _cache is None or len(_cache) < 20:
            return "HOLD"

        close = _cache['Close'].values
        sma = np.mean(close[-20:])
        current = close[-1]

        if current < sma * 0.98:
            return "BUY"
        elif current > sma * 1.02:
            return "SELL"
        else:
            return "HOLD"
    except Exception:
        return "HOLD"
"""
    is_valid, msg = _validate_generated_code(valid_code, "test-valid")
    print(f"Valid: {is_valid}")
    print(f"Message:\n{msg}")
    assert is_valid, "Should accept valid code"
    print("✅ PASS - Correctly accepted valid code")

    # Test 6: Code with warnings (acceptable but not perfect)
    print("\n6. Testing code with WARNINGS (acceptable):")
    print("-" * 70)
    warning_code = """
import yfinance as yf
import numpy as np

def execute_trade(ticker, cash_balance, shares_held):
    # No try/except - will generate warning
    data = yf.download(ticker, start="2023-01-01", end="2007-10-09", progress=False)

    if len(data) > 20:
        close = data['Close'].values
        if close[-1] > np.mean(close[-20:]):
            return "BUY"

    return "HOLD"
"""
    is_valid, msg = _validate_generated_code(warning_code, "test-warning")
    print(f"Valid: {is_valid}")
    print(f"Message:\n{msg}")
    assert is_valid, "Should accept code with warnings"
    print("✅ PASS - Correctly accepted code with warnings")

    # Test actual generated algorithms
    print("\n7. Testing ACTUAL generated algorithms:")
    print("-" * 70)

    algo_dir = Path(__file__).parent / "generate_algo"
    if algo_dir.exists():
        for algo_file in algo_dir.glob("generated_algo_*.py"):
            print(f"\nTesting: {algo_file.name}")
            with open(algo_file, 'r') as f:
                code = f.read()

            is_valid, msg = _validate_generated_code(code, algo_file.stem)

            status = "✅ PASS" if is_valid else "❌ FAIL"
            print(f"{status} - {algo_file.name}")

            if not is_valid:
                print(f"   Reason: {msg.split(chr(10))[0]}")
            elif "WARNING" in msg:
                print(f"   {msg.split(chr(10))[1] if len(msg.split(chr(10))) > 1 else 'Has warnings'}")
    else:
        print("⚠️ No generated algorithms found to test")

    print("\n" + "="*70)
    print("ALL VALIDATION TESTS PASSED! ✅")
    print("="*70)

if __name__ == "__main__":
    test_validation()
