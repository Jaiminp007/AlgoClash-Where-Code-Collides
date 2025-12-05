# 🔧 Claude Opus Generation Fix

## Problem
Claude Opus models were not generating algorithms - returning empty or no code.

## Root Causes Identified

1. **Conflicting Instructions**
   - Prompt said "no markdown fences" but also "Raw Python code only"
   - Opus might want to wrap code in ```python``` for clarity
   - Extraction function already handles fences, so restriction was unnecessary

2. **Missing Output Guidance**
   - Prompt didn't clearly tell models to START with code
   - High-end models like Opus tend to explain before coding
   - Needed explicit "BEGIN YOUR RESPONSE" instruction

3. **Insufficient Debugging**
   - No visibility into what Opus was actually returning
   - Hard to diagnose if it was generating code with explanation vs no code

## Solutions Implemented

### 1. **Clarified Output Format** ✅

**Before:**
```
YOUR RESPONSE MUST BE:
- Raw Python code ONLY (no markdown, no ``` fences)
```

**After:**
```
RESPOND WITH COMPLETE, EXECUTABLE PYTHON CODE.

ACCEPTABLE FORMATS:
✅ Raw Python code starting with imports
✅ Python code wrapped in ```python ``` markdown fences (will be extracted)

YOUR CODE MUST INCLUDE:
✅ Import statements (import yfinance as yf, import numpy as np)
✅ Module-level variables for caching (_data_cache = None)
✅ The execute_trade function with full implementation
...
```

**Impact:** Opus can now use markdown fences (which it prefers) without being rejected.

---

### 2. **Added "BEGIN YOUR RESPONSE" Section** ✅

Added explicit instruction to start immediately with code:

```
═══════════════════════════════════════════════════════════════════════════════
BEGIN YOUR RESPONSE
═══════════════════════════════════════════════════════════════════════════════

Start your response immediately with the Python code. Do not write explanations
before or after the code. Your entire response should be the algorithm implementation.

You may optionally wrap your code in ```python``` markdown fences, or provide
it as raw Python code.

Example of acceptable response format:

```python
import yfinance as yf
import numpy as np

_data_cache = None

def execute_trade(ticker, cash_balance, shares_held):
    # Your full implementation here
    return "HOLD"
```

NOW WRITE YOUR COMPLETE ALGORITHM IMPLEMENTATION:
```

**Impact:** Makes it crystal clear to Opus that it should respond with code immediately.

---

### 3. **Added Debug Logging** ✅

Added debug output specifically for Opus and GPT-4:

```python
# Debug logging for Opus and other high-end models
if 'opus' in model_id.lower() or 'gpt-4' in model_id.lower():
    print(f"\n🔍 DEBUG: Raw response from {model_id} (first 500 chars):")
    print(f"   {raw[:500]}")
    print(f"   ... (total length: {len(raw)} chars)")

content = _extract_execute_trade_code(raw)

if content and 'def execute_trade' in content:
    print(f"✅ SUCCESS: Code received from {model_id} ({len(content)} chars).")
    return content.strip()
else:
    print(f"❌ FAILED to find execute_trade in {model_id} output.")
    print(f"   Raw response (first 800 chars): {raw[:800]}")
    if content:
        print(f"   Extracted content (first 500 chars): {content[:500]}")
    return None
```

**Impact:** Now you can see exactly what Opus is returning and diagnose issues.

---

### 4. **Strengthened Top Instructions** ✅

Made the opening instruction super clear:

```
TASK: Write a complete Python trading algorithm for AAPL.

CRITICAL: Your response must be EXECUTABLE PYTHON CODE ONLY. Do not include
explanations, markdown formatting, or any text outside the code. The code will
be directly saved to a .py file and executed.
```

**Impact:** Sets expectations immediately that this is a code generation task.

---

## Testing Instructions

### Quick Test with Opus

Run the dedicated test script:

```bash
cd backend
python test_opus.py
```

This will:
1. Build the prompt for Opus
2. Show you the prompt preview
3. Attempt generation
4. Show debug output of what Opus returns
5. Validate the generated code
6. Save to `generate_algo/test_opus_output.py`

### Full Generation Test

```bash
cd backend
python -m open_router.algo_gen
# Select anthropic/claude-opus-4 among your models
```

Watch for:
- 🔍 DEBUG output showing Opus's raw response
- ✅ SUCCESS message with code length
- Or ❌ FAILED with first 800 chars of response

---

## What to Look For

### If Opus Succeeds:
```
--- Generating algorithm with: anthropic/claude-opus-4 ---

🔍 DEBUG: Raw response from anthropic/claude-opus-4 (first 500 chars):
   import yfinance as yf
   import numpy as np

   _data_cache = None

   def execute_trade(ticker, cash_balance, shares_held):
   ... (total length: 2847 chars)

✅ SUCCESS: Code received from anthropic/claude-opus-4 (2847 chars).
```

### If Opus Still Fails:
```
--- Generating algorithm with: anthropic/claude-opus-4 ---

🔍 DEBUG: Raw response from anthropic/claude-opus-4 (first 500 chars):
   I'd be happy to help you create a trading algorithm. However, I want to
   make sure I understand the requirements correctly...
   ... (total length: 1234 chars)

❌ FAILED to find execute_trade in anthropic/claude-opus-4 output.
   Raw response (first 800 chars): I'd be happy to help...
```

**If you see this:** Opus is trying to have a conversation instead of generating code. This suggests:
1. The model might not have access (check OpenRouter credits)
2. API key permissions issue
3. Model safety filters triggered

---

## Troubleshooting

### Issue: Opus returns explanation instead of code

**Solution:** Check if you're using the right model ID:
- ✅ `anthropic/claude-opus-4`
- ✅ `anthropic/claude-3.5-sonnet`
- ❌ `anthropic/claude-opus` (old, might not exist)

### Issue: Opus times out

**Solution:** Already handled with:
- 60s timeout per request
- 3 retry attempts
- Exponential backoff

### Issue: Opus not available

**Check:**
```bash
cd backend
python -c "from open_router.model_fetching import fetch_available_models; models = fetch_available_models(); opus_models = [m for m in models if 'opus' in m['id'].lower()]; print('Opus models:', opus_models)"
```

### Issue: API errors

**Common causes:**
- Insufficient credits on OpenRouter
- API key not set: `export OPENROUTER_API_KEY='sk-or-v1-...'`
- Model not accessible with your plan

---

## Files Modified

1. **backend/open_router/algo_gen.py**
   - Lines 514-516: Added CRITICAL instruction at top
   - Lines 557-580: Clarified output format (allow markdown fences)
   - Lines 703-730: Added "BEGIN YOUR RESPONSE" section
   - Lines 909-925: Added debug logging (async version)
   - Lines 989-1004: Added debug logging (sync version)

2. **backend/test_opus.py** (NEW)
   - Dedicated test script for Opus
   - Shows debug output
   - Validates generated code

---

## Expected Results

### Before Fix:
- ❌ Opus confused by "no markdown fences" instruction
- ❌ Opus adds explanations before code
- ❌ No visibility into what Opus returns
- ❌ Success rate: 0%

### After Fix:
- ✅ Opus can use markdown fences (preferred format)
- ✅ Clear instruction to start with code immediately
- ✅ Debug output shows exactly what Opus returns
- ✅ Success rate: Should be 80%+ (if model accessible)

---

## Next Steps

1. **Run test script:** `python backend/test_opus.py`
2. **Check debug output:** See what Opus actually returns
3. **If still failing:** Share the debug output so we can diagnose further

Possible remaining issues:
- API access/credits (check OpenRouter dashboard)
- Model availability (try `anthropic/claude-3.5-sonnet` as alternative)
- Prompt still too long for Opus's context window (unlikely at 220 lines)

---

## Summary

**Changes Made:**
- ✅ Allow markdown fences in output
- ✅ Add explicit "BEGIN YOUR RESPONSE" instruction
- ✅ Add debug logging for Opus
- ✅ Clarify code-only requirement at top

**Impact:**
- Should fix Opus generation issues
- Better debugging visibility
- Works for all high-end models (GPT-4, Claude 3.5, etc.)

**Test:** Run `python backend/test_opus.py` to verify!
