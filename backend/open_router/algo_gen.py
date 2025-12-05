import os
import requests
import json
from dotenv import load_dotenv
import time
import hashlib
import random
import asyncio
import httpx
from typing import Optional, List, Tuple, Dict

# Import the main function from your model fetching script
# Use relative import so this works when imported as part of the package (open_router)
from .model_fecthing import get_models_to_use
# Load environment variables from a .env file (explicit backend path for reliability)
try:
    # First, try default discovery (current CWD and parents)
    load_dotenv()
    # Then, explicitly load backend/.env relative to this file
    _backend_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)
except Exception:
    pass

# --- 1. Configuration ---
API_KEY = os.getenv('OPENROUTER_API_KEY')
CHAT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_API_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "ai-trader-battlefield")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'generate_algo')

def _wrap_code_if_missing_func(code: str) -> str:
    """Ensure the returned code defines execute_trade; wrap if missing."""
    if code and 'def execute_trade' not in code:
        return (
            "def execute_trade(ticker, cash_balance, shares_held):\n"
            "    # Wrapped fallback if model omitted function signature\n"
            "    try:\n"
            "        pass\n"
            "    except Exception:\n"
            "        return 'HOLD'\n"
            "    return 'HOLD'\n"
        )
    return code


def _generate_fallback_code(ticker: str, model_id: str) -> str:
    """Produce a safe, diversified stateful algorithm when API is unavailable.
    Uses position-based logic without fetching market data.
    """
    seed_int = int(hashlib.md5(model_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_int)

    # Generate diverse parameters for each agent
    cycle_length = rng.choice([5, 10, 15, 20, 25, 30])
    buy_threshold = rng.randint(2, 8)
    sell_threshold = rng.randint(10, 20)
    target_position = rng.choice([5, 10, 15, 20])

    state_var = f"_state_{hashlib.md5(model_id.encode()).hexdigest()[:6]}"

    code = [
        f"{state_var} = {{'cycle': 0, 'target': {target_position}, 'direction': 1}}",
        "",
        "def execute_trade(ticker, cash_balance, shares_held):",
        f"    global {state_var}",
        "    try:",
        f"        {state_var}['cycle'] += 1",
        f"        cycle = {state_var}['cycle']",
        "",
        "        # Position-based strategy without market data",
        f"        if cycle % {cycle_length} == 0:",
        f"            {state_var}['direction'] *= -1",
        "",
        f"        target = {state_var}['target'] * {state_var}['direction']",
        "",
        f"        if shares_held < {buy_threshold} and cash_balance > 1000:",
        "            return 'BUY'",
        f"        elif shares_held > {sell_threshold}:",
        "            return 'SELL'",
        "        elif shares_held < target:",
        "            return 'BUY'",
        "        elif shares_held > target:",
        "            return 'SELL'",
        "        return 'HOLD'",
        "    except Exception:",
        "        return 'HOLD'",
    ]

    return "\n".join(code) + "\n"


def _prepend_historical_data(code: str, compressed_data: Optional[Dict[str, List]]) -> str:
    """Prepend embedded MongoDB historical arrays so generated code can use them directly."""
    if not compressed_data:
        return code

    data_header = f"""import numpy as np

# EMBEDDED HISTORICAL DATA (Auto-generated from MongoDB)
# Total: {len(compressed_data['closes'])} trading days
# Date range: {compressed_data['dates'][0]} to {compressed_data['dates'][-1]}

_HISTORICAL_DATES = {compressed_data['dates']}
_HISTORICAL_OPENS = {compressed_data['opens']}
_HISTORICAL_HIGHS = {compressed_data['highs']}
_HISTORICAL_LOWS = {compressed_data['lows']}
_HISTORICAL_CLOSES = {compressed_data['closes']}
_HISTORICAL_VOLUMES = {compressed_data['volumes']}

# --- Generated Algorithm Code Below ---

"""

    # Strip duplicate imports or attempts to redefine the arrays
    code_lines = code.split('\n')
    filtered_lines: List[str] = []
    skipping_array = False

    for line in code_lines:
        stripped = line.strip()

        if stripped.startswith('import numpy') or stripped.startswith('from numpy'):
            continue

        if stripped.startswith('_HISTORICAL_') and '=' in stripped:
            skipping_array = True
            continue

        if skipping_array:
            if ']' in line:
                skipping_array = False
            continue

        filtered_lines.append(line)

    clean_code = '\n'.join(filtered_lines)
    return data_header + clean_code


def load_csv_preview(*_args, **_kwargs) -> str:
    """Backwards-compatible stub for legacy tests that patch this helper."""
    return ""


# Simple in-memory cache for prompt context (avoids repeated MongoDB fetches)
_prompt_context_cache: Dict[str, Tuple] = {}


def fetch_prompt_context(ticker: str, end_date: str) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict], str]:
    """Load MongoDB analysis + compressed data + extracted patterns for prompts. 
    Returns (analysis, compressed, patterns, source).
    Results are cached in memory to avoid repeated MongoDB fetches.
    """
    cache_key = f"{ticker}_{end_date}"
    if cache_key in _prompt_context_cache:
        return _prompt_context_cache[cache_key]
    
    analysis = None
    compressed = None
    patterns = None
    source = "fallback"

    try:
        import sys
        backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from mongodb_analysis import (  # type: ignore
            analyze_mongodb_data, 
            generate_compressed_historical, 
            extract_real_patterns,
            get_analysis_data,
            format_patterns_for_prompt
        )

        analysis = analyze_mongodb_data(ticker, end_date=end_date)
        compressed = generate_compressed_historical(ticker, end_date=end_date)
        patterns = extract_real_patterns(ticker, end_date=end_date)
        
        # Also get the enhanced analysis data
        enhanced_analysis = get_analysis_data(ticker)
        if enhanced_analysis:
            # Add formatted patterns to the patterns dict for easy prompt embedding
            patterns = patterns or {}
            patterns['enhanced_patterns'] = enhanced_analysis.get('patterns', {})
            patterns['formatted_patterns'] = format_patterns_for_prompt(enhanced_analysis.get('patterns', {}))
            print(f"✅ Enhanced pattern analysis loaded for {ticker}")
        
        if analysis and compressed:
            source = "mongodb"
        else:
            print(f"⚠️ MongoDB context unavailable for {ticker}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"⚠️ Failed to load MongoDB context for {ticker}: {exc}")

    result = (analysis, compressed, patterns, source)
    _prompt_context_cache[cache_key] = result
    return result


def _save_code_for_model(code: str, model_name: str):
    safe_name = model_name.replace('/', '_').replace('-', '_').replace(':', '_').replace('.', '_')
    save_algorithm_to_file(code, safe_name)


async def _generate_algorithms_for_agents_async(
    selected_agents: List[str],
    ticker: str,
    progress_callback=None
) -> bool:
    """
    Async implementation: Generate algorithms for specific agents concurrently.
    Returns True if at least one agent produced valid code containing execute_trade.
    """
    total = len(selected_agents or [])
    print(f"[gen] Generating algorithms for {total} agents using {ticker} data (ASYNC)")
    print(f"[ok] Using {total} models selected from frontend")

    if not selected_agents:
        print("❌ No models provided for generation")
        if progress_callback:
            progress_callback(35, "No models provided for generation")
        return False

    base_prompt_data = build_generation_prompt(ticker)
    if isinstance(base_prompt_data, tuple):
        base_prompt, compressed_data = base_prompt_data
    else:
        base_prompt, compressed_data = base_prompt_data, None

    # Create lightweight prompt for fallback
    lightweight_data = build_generation_prompt(ticker, lightweight=True)
    if isinstance(lightweight_data, tuple):
        lightweight_prompt = lightweight_data[0]
    else:
        lightweight_prompt = lightweight_data

    api_available = bool(API_KEY)
    if not api_available:
        msg = "OPENROUTER_API_KEY not found. Cannot generate algorithms."
        print(f"❌ {msg}")
        if progress_callback:
            progress_callback(40, msg)
        return False

    # Preflight: fetch accessible models and normalize IDs to avoid attempting gated models
    accessible = _get_accessible_models()
    if accessible is None:
        warn_msg = "Could not query accessible models from OpenRouter; proceeding without prefilter. Some models may be unavailable."
        print(f"⚠️ {warn_msg}")
        if progress_callback:
            progress_callback(42, warn_msg)
        accessible = set()
    normalized_accessible = _normalize_model_id_set(accessible)

    # Remap selected agents to accessible IDs when possible (toggle :free suffix as needed)
    remapped_agents = []
    filtered_out = []
    for m in selected_agents:
        mm = _pick_best_accessible_id(m, normalized_accessible)
        if mm is None and normalized_accessible:
            filtered_out.append(m)
        else:
            remapped_agents.append(mm or m)

    if filtered_out:
        msg = (
            f"Skipping {len(filtered_out)} unavailable model(s): "
            + ", ".join(filtered_out)
        )
        print(f"⚠️ {msg}")
        if progress_callback:
            progress_callback(45, msg)
            # Emit per-model skip events for UI
            for m in filtered_out:
                try:
                    progress_callback(45, f"MODEL_SKIP::{m}::unavailable")
                except Exception:
                    pass

    # If all models filtered, fail early
    if not remapped_agents:
        msg = "No accessible models available with current API key."
        print(f"❌ {msg}")
        if progress_callback:
            progress_callback(48, msg)
        return False

    # Update total after filtering
    total = len(remapped_agents)

    # Create httpx async client with connection pooling
    timeout = httpx.Timeout(60.0, connect=10.0)
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        # Create tasks for all agents with their original index to preserve order
        async def generate_with_index(index: int, agent_model: str) -> Tuple[int, str, Optional[str]]:
            """Generate algorithm and return (index, model_id, code) to preserve order."""
            try:
                step_prog = 30 + int((index / max(1, total)) * 25)  # distribute 30-55%
                if progress_callback:
                    # Emit a per-model start event so UI can mark it as actively generating (supports concurrency)
                    try:
                        progress_callback(step_prog, f"MODEL_START::{agent_model}")
                    except Exception:
                        pass
                    # Retain the existing human-readable message for backwards compatibility
                    try:
                        progress_callback(step_prog, f"Generating algorithm {index+1}/{total} using {agent_model}...")
                    except Exception:
                        pass
                print(f"\nGenerating algorithm {index+1}/{total} using {agent_model}...")

                per_model_prompt = base_prompt + build_diversity_directives(agent_model)
                per_model_fallback = lightweight_prompt + build_diversity_directives(agent_model)
                code = await generate_algorithm_async(client, agent_model, per_model_prompt, fallback_prompt_text=per_model_fallback)

                if code:
                    code = _prepend_historical_data(code, compressed_data)

                # Require a proper execute_trade from the model
                if not code or 'def execute_trade' not in code:
                    print(f"[error] Missing valid execute_trade in response for {agent_model}")
                    if progress_callback:
                        try:
                            progress_callback(step_prog, f"MODEL_FAIL::{agent_model}::missing_execute_trade")
                        except Exception:
                            pass
                    return (index, agent_model, None)

                # Emit a short preview snippet to the API progress stream for UX
                if progress_callback and code:
                    try:
                        snippet_lines = code.splitlines()[:24]
                        preview = "\n".join(snippet_lines)
                        progress_callback(step_prog, f"PREVIEW::{agent_model}::{preview}")
                    except Exception:
                        pass

                if progress_callback:
                    try:
                        progress_callback(step_prog, f"MODEL_OK::{agent_model}::generated")
                    except Exception:
                        pass

                return (index, agent_model, code)

            except Exception as e:
                print(f"[error] Error generating algorithm for {agent_model}: {e}")
                if progress_callback:
                    try:
                        progress_callback(30, f"MODEL_FAIL::{agent_model}::{str(e)}")
                    except Exception:
                        pass
                return (index, agent_model, None)

        # Use semaphore to limit concurrent requests (max 6 as specified)
        semaphore = asyncio.Semaphore(6)

        async def generate_with_semaphore(index: int, agent_model: str):
            async with semaphore:
                return await generate_with_index(index, agent_model)

        # Fire all requests concurrently
        tasks = [
            generate_with_semaphore(i, agent_model)
            for i, agent_model in enumerate(remapped_agents)
        ]

        # Wait for all with overall timeout guard (60s per request * 6 + buffer)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=False),
                timeout=400.0  # Overall timeout
            )
        except asyncio.TimeoutError:
            print("❌ Overall timeout exceeded for algorithm generation")
            if progress_callback:
                progress_callback(55, "Algorithm generation timeout exceeded")
            return False

    # Process results in original order
    failures = []
    saved = 0

    for index, agent_model, code in sorted(results, key=lambda x: x[0]):
        if code:
            _save_code_for_model(code, agent_model)
            saved += 1
            if progress_callback:
                try:
                    progress_callback(30 + int((index / max(1, total)) * 25), f"MODEL_OK::{agent_model}::saved")
                except Exception:
                    pass
        else:
            failures.append(agent_model)

    # Only fail if NO algorithms were generated successfully
    if saved == 0:
        msg = "Algorithm generation completely failed - no valid algorithms generated"
        print(f"❌ {msg}")
        if progress_callback:
            progress_callback(55, msg)
        return False

    # Warn about failures but continue with successful ones
    if failures:
        msg = f"⚠️ Algorithm generation failed for {len(failures)} model(s): {', '.join(failures)}. Continuing with {saved} successful algorithms."
        print(msg)
        if progress_callback:
            progress_callback(55, msg)
    else:
        if progress_callback:
            progress_callback(55, "All algorithms generated successfully!")

    print(f"\n[done] Algorithm generation completed for {ticker} ({saved}/{total} successful)")
    return True


def generate_algorithms_for_agents(selected_agents, ticker, progress_callback=None):
    """
    Sync wrapper: Generate algorithms for specific agents via API only (no local fallbacks).
    Returns True only if all agents produced valid code containing execute_trade.

    This is a synchronous wrapper around the async implementation that preserves
    the original function signature for backward compatibility.
    """
    # Run the async version in a new event loop
    return asyncio.run(_generate_algorithms_for_agents_async(selected_agents, ticker, progress_callback))

def build_generation_prompt(ticker: str, lightweight: bool = False) -> Tuple[str, Optional[Dict]]:
    """Build a comprehensive prompt for REAL quantitative trading algorithms."""

    # Determine simulation start from MongoDB context
    first_date = "2025-11-24"
    
    analysis, compressed, patterns, source = fetch_prompt_context(ticker, first_date)

    # Build market context section
    market_context = ""
    pattern_context = ""
    trend_direction = "UNKNOWN"
    
    if analysis and compressed:
        trend_direction = analysis['trends']['direction']
        current_price = analysis['price_stats']['current']
        rsi = analysis['momentum']['rsi_14']
        
        market_context = f"""
## MARKET DATA FOR {ticker}
- Current Price: ${current_price:.2f}
- Trend: {trend_direction} (Strength: {analysis['trends']['strength']}/10)
- RSI(14): {rsi:.1f}
- Volatility Regime: {analysis['volatility']['regime']}
- SMA(20): ${analysis['trends']['sma_20']:.2f}
- SMA(50): ${analysis['trends']['sma_50']:.2f}
- Bollinger Upper: ${analysis['volatility']['bb_upper']:.2f}
- Bollinger Lower: ${analysis['volatility']['bb_lower']:.2f}
- Support: ${analysis['levels']['support_strong']:.2f}
- Resistance: ${analysis['levels']['resistance_strong']:.2f}
"""
    
    # Build pattern context section from extracted patterns
    if patterns:
        total_mins = patterns.get('total_minutes', 0)
        mean_rev = patterns.get('mean_reversion_rate', 0.5)
        momentum_pers = patterns.get('momentum_persistence', 0.5)
        
        # Determine strategy hint based on patterns
        if mean_rev > 0.55:
            strategy_from_patterns = "MEAN_REVERSION"
            pattern_advice = "Use RSI oversold/overbought signals. Buy when RSI < 30, sell when RSI > 70"
        elif momentum_pers > 0.55:
            strategy_from_patterns = "MOMENTUM"
            pattern_advice = "Use moving average crossovers. Buy when EMA(12) > EMA(26), sell when EMA(12) < EMA(26)"
        else:
            strategy_from_patterns = "HYBRID"
            pattern_advice = "Combine RSI for entry timing with MA crossovers for trend confirmation"
        
        pattern_context = f"""
## STATISTICAL PATTERNS FROM {total_mins:,} MINUTES OF HISTORICAL DATA

- Mean Reversion Rate: {mean_rev:.1%} (after 0.5% moves)
- Momentum Persistence: {momentum_pers:.1%}
- **RECOMMENDED APPROACH: {strategy_from_patterns}**
- **IMPLEMENTATION: {pattern_advice}**

{patterns.get('formatted_patterns', '')}
"""

    base = f'''You are an elite HIGH-FREQUENCY SCALPING TRADER at a top hedge fund. Write an aggressive intraday trading algorithm for {ticker}.

## YOUR MISSION: ACHIEVE 5%+ ROI IN ONE TRADING DAY

The market only moves ~2% per day, but you have:
- **4x LEVERAGE** (can trade up to $40,000 with $10,000 capital)
- **SHORT SELLING** (profit from both up AND down moves)
- **390 TICKS** (one full trading day of minute data)

**THE KEY TO 5%+ ROI:** Trade FREQUENTLY with LEVERAGE!
- Make 15-30 trades per day
- Target 0.3-0.5% profit per trade
- Use 300-400 shares per trade (leverage!)
- Cut losses fast at -0.3%

## YOUR ROLE
You have expertise in:
- High-frequency scalping strategies
- Momentum and mean reversion on minute timeframes
- Aggressive position sizing with leverage
- Tight stop-losses and quick profit-taking

## AVAILABLE HISTORICAL DATA (MUST USE!)
Your algorithm has access to these pre-loaded arrays:
```python
_HISTORICAL_DATES   # ['2025-01-02', '2025-01-03', ...]  - 217 trading days
_HISTORICAL_OPENS   # [248.92, 243.37, ...]
_HISTORICAL_HIGHS   # [249.10, 244.17, ...]
_HISTORICAL_LOWS    # [241.82, 241.89, ...]
_HISTORICAL_CLOSES  # [243.82, 243.30, ...]
_HISTORICAL_VOLUMES # [48004805, 31628615, ...]
```

**YOU MUST USE THIS DATA** to calculate support/resistance levels, historical volatility, and inform your signals.

{market_context}
{pattern_context}

## REQUIREMENTS - REAL QUANT ALGORITHM

### 1. MUST INCLUDE THESE HELPER FUNCTIONS:

```python
def calculate_sma(prices, period):
    """Simple Moving Average"""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

def calculate_ema(prices, period):
    """Exponential Moving Average"""
    if len(prices) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_rsi(prices, period=14):
    """Relative Strength Index"""
    if len(prices) < period + 1:
        return 50
    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(0, change))
        losses.append(max(0, -change))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Bollinger Bands - returns (lower, middle, upper)"""
    if len(prices) < period:
        return None, None, None
    sma = sum(prices[-period:]) / period
    variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
    std = variance ** 0.5
    return sma - std_dev * std, sma, sma + std_dev * std
```

### 2. MUST HAVE THESE COMPONENTS IN execute_trade():

a) **Technical Indicator Calculation**:
   - Calculate SMA(20), SMA(50) or EMA(12), EMA(26)
   - Calculate RSI(14)
   - Calculate Bollinger Bands

b) **Entry Signals** (TRADE FREQUENTLY for small gains):
   - BUY when price dips 0.2-0.3% from recent peak AND RSI < 45
   - SELL/SHORT when price rises 0.2-0.3% from recent low AND RSI > 55

c) **Position Tracking**:
   - Track entry price in a global variable
   - Calculate P&L percentage for stop-loss/take-profit

d) **Risk Management (TIGHT for scalping)**:
   - Stop-loss at -0.3% from entry (cut losses fast!)
   - Take-profit at +0.5% from entry (take profits quickly!)
   - Maximum position: 350 shares (use leverage!)

e) **Exit Logic**:
   - Close all positions before tick 380 (end of trading day)
   - Time-based exit: Don't hold more than 25 ticks

### 3. FUNCTION SIGNATURE (EXACT):
```python
def execute_trade(ticker, price, tick, cash_balance, shares_held):
    """
    Args:
        ticker: Stock symbol (str)
        price: Current price (float)
        tick: Current tick 0-389 (int)
        cash_balance: Available cash, can be negative for margin (float)
        shares_held: Current position, negative = short (int)
    
    Returns:
        ("BUY", quantity) or ("SELL", quantity) or "HOLD"
    """
```

## COMPLETE EXAMPLE OF A PROPER SCALPING ALGORITHM:

```python
# Global state for aggressive scalping
_prices = []
_entry_price = None
_entry_tick = None
_position_type = None
_trade_count = 0
_recent_high = None
_recent_low = None

def calculate_sma(prices, period):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(0, change))
        losses.append(max(0, -change))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def execute_trade(ticker, price, tick, cash_balance, shares_held):
    global _prices, _entry_price, _entry_tick, _position_type, _trade_count
    global _recent_high, _recent_low
    
    _prices.append(price)
    
    # Track recent high/low for scalping signals
    if len(_prices) >= 10:
        _recent_high = max(_prices[-10:])
        _recent_low = min(_prices[-10:])
    else:
        _recent_high = price
        _recent_low = price
    
    # Need minimal data for indicators
    if tick < 15:
        return "HOLD"
    
    # Calculate indicators
    sma_10 = calculate_sma(_prices, 10)
    rsi = calculate_rsi(_prices, 10)  # Faster RSI for scalping
    
    # STOP-LOSS / TAKE-PROFIT for existing position (TIGHT for scalping)
    if shares_held != 0 and _entry_price:
        if shares_held > 0:  # Long position
            pnl_pct = (price - _entry_price) / _entry_price * 100
        else:  # Short position
            pnl_pct = (_entry_price - price) / _entry_price * 100
        
        ticks_held = tick - _entry_tick if _entry_tick else 0
        
        # Stop-loss: -0.3% (cut losses fast!)
        if pnl_pct <= -0.3:
            _entry_price = None
            _entry_tick = None
            _position_type = None
            if shares_held > 0:
                return ("SELL", shares_held)
            else:
                return ("BUY", abs(shares_held))
        
        # Take-profit: +0.5% (take profits quickly!)
        if pnl_pct >= 0.5:
            _entry_price = None
            _entry_tick = None
            _position_type = None
            _trade_count += 1
            if shares_held > 0:
                return ("SELL", shares_held)
            else:
                return ("BUY", abs(shares_held))
        
        # Time-based exit: Don't hold more than 25 ticks
        if ticks_held >= 25:
            _entry_price = None
            _entry_tick = None
            _position_type = None
            if shares_held > 0:
                return ("SELL", shares_held)
            else:
                return ("BUY", abs(shares_held))
    
    # CLOSE all positions near end of day
    if tick >= 370:
        if shares_held > 0:
            return ("SELL", shares_held)
        elif shares_held < 0:
            return ("BUY", abs(shares_held))
        return "HOLD"
    
    # SCALPING ENTRY SIGNALS (trade frequently!)
    
    # Calculate price deviation from recent range
    price_from_high = (price - _recent_high) / _recent_high * 100 if _recent_high else 0
    price_from_low = (price - _recent_low) / _recent_low * 100 if _recent_low else 0
    
    # BUY signal: Price dipped from recent high + RSI not overbought
    buy_signal = price_from_high < -0.2 and rsi < 55
    
    # SELL signal: Price rose from recent low + RSI not oversold  
    sell_signal = price_from_low > 0.2 and rsi > 45
    
    # Execute trades with LEVERAGE (300 shares)
    trade_size = 300
    
    if shares_held == 0:  # No position - look for entry
        if buy_signal:
            _entry_price = price
            _entry_tick = tick
            _position_type = 'long'
            return ("BUY", trade_size)
        elif sell_signal:
            _entry_price = price
            _entry_tick = tick
            _position_type = 'short'
            return ("SELL", trade_size)
    
    return "HOLD"
```

## OUTPUT REQUIREMENTS

1. Output ONLY valid Python code
2. Include the helper functions (calculate_sma, calculate_rsi, etc.)
3. Include global state variables (_prices, _entry_price, etc.)
4. Include the execute_trade function with the exact signature
5. NO markdown code fences, NO explanations, NO comments about what you're doing
6. The code must be syntactically correct Python

Start your output with the global variable declarations.
'''

    return base, compressed


def build_diversity_directives(model_id: str) -> str:
    """Create diverse QUANTITATIVE strategy directives for each model."""
    # Deterministic seed from model_id
    seed_int = int(hashlib.md5(model_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_int)

    # Real quantitative strategies - AGGRESSIVE INTRADAY SCALPING for 5%+ ROI
    # With 4x leverage and frequent trading, we target many small wins
    strategies = [
        ("RSI_SCALPER", """
## YOUR STRATEGY: RSI Scalper (HIGH FREQUENCY)

**GOAL: Achieve 5%+ ROI through frequent small wins with leverage**

**LEVERAGE: You have 4x margin - USE IT! Trade 200-400 shares per position**

**ENTRY SIGNALS (trade FAST, target 0.3-0.5% per trade):**
- BUY when RSI(14) < 40 AND price dips 0.2% from recent high
- SELL/SHORT when RSI(14) > 60 AND price rises 0.2% from recent low

**EXIT SIGNALS (quick profits):**
- Take-profit: +0.5% from entry (don't wait for bigger moves!)
- Stop-loss: -0.3% from entry (cut losses fast)
- Maximum hold: 20 ticks

**POSITION SIZING:**
- Trade 300 shares per position (use leverage!)
- Exit and re-enter frequently - aim for 15-30 trades per day
"""),
        ("MOMENTUM_SCALPER", """
## YOUR STRATEGY: Momentum Scalper (HIGH FREQUENCY)

**GOAL: Achieve 5%+ ROI by riding micro-trends with leverage**

**LEVERAGE: You have 4x margin - USE IT! Trade 200-400 shares per position**

**ENTRY SIGNALS:**
- BUY when price rises 0.15% in last 3 ticks (momentum starting)
- SELL/SHORT when price falls 0.15% in last 3 ticks

**EXIT SIGNALS:**
- Take-profit: +0.4% from entry
- Stop-loss: -0.25% from entry
- Exit if momentum reverses (opposite 0.1% move)

**KEY: Trade with the micro-trend, not against it!**
- Track price changes over last 5 ticks
- Use 300-400 shares per trade
"""),
        ("BOLLINGER_SCALPER", """
## YOUR STRATEGY: Bollinger Band Scalper (HIGH FREQUENCY)

**GOAL: Achieve 5%+ ROI by mean reversion with leverage**

**LEVERAGE: You have 4x margin - USE IT! Trade 200-400 shares**

**ENTRY SIGNALS:**
- BUY when price touches lower Bollinger Band (use 1.5 std dev for tighter bands)
- SELL/SHORT when price touches upper Bollinger Band

**EXIT SIGNALS:**
- Take-profit: Price returns to middle band (SMA) = ~0.4% gain
- Stop-loss: -0.3% from entry
- Time stop: Exit after 15 ticks regardless

**PARAMETERS:**
- Use Bollinger(10, 1.5) for faster signals (not 20, 2)
- Trade 350 shares per position
"""),
        ("VWAP_SCALPER", """
## YOUR STRATEGY: VWAP Scalper (HIGH FREQUENCY)

**GOAL: Achieve 5%+ ROI by trading around fair value**

**LEVERAGE: You have 4x margin - Trade 300 shares per position**

**CALCULATE VWAP:**
- Track cumulative (price * volume) / cumulative volume
- Or approximate with SMA(20) as fair value proxy

**ENTRY SIGNALS:**
- BUY when price drops 0.3% below VWAP/SMA(20)
- SELL/SHORT when price rises 0.3% above VWAP/SMA(20)

**EXIT SIGNALS:**
- Take-profit: Price returns to VWAP (+0.3% gain)
- Stop-loss: -0.25% from entry

**TRADE FREQUENTLY: 20+ trades per day!**
"""),
        ("BREAKOUT_SCALPER", """
## YOUR STRATEGY: Micro-Breakout Scalper (HIGH FREQUENCY)

**GOAL: Achieve 5%+ ROI by catching small breakouts**

**LEVERAGE: You have 4x margin - Trade 300-400 shares**

**TRACK 10-TICK RANGE:**
- High10 = max price of last 10 ticks
- Low10 = min price of last 10 ticks

**ENTRY SIGNALS:**
- BUY when price breaks above High10 by 0.1%
- SELL/SHORT when price breaks below Low10 by 0.1%

**EXIT SIGNALS:**
- Take-profit: +0.4% from entry (ride the breakout)
- Stop-loss: -0.2% from entry (false breakout)
- Time stop: 10 ticks max hold

**KEY: Small range = more breakouts = more trades = more profit!**
"""),
    ]

    strategy = rng.choice(strategies)
    name, directive = strategy

    return f'''
{directive}

## REQUIRED CODE STRUCTURE:

Include these helper functions:
- calculate_sma(prices, period)
- calculate_ema(prices, period)
- calculate_rsi(prices, period=14)
- calculate_bollinger_bands(prices, period=20, std_dev=2)

Include global state:
- _prices = []
- _entry_price = None
- _position_type = None

Include proper execute_trade function with:
- Indicator calculations
- Entry/exit logic based on strategy above
- Stop-loss and take-profit
- Position tracking
'''


# =============================================================================
# ADAPTATION HELPERS - For mid-simulation algorithm improvement
# =============================================================================

def _format_recent_trades(trades: list) -> str:
    """Format recent trades for the adaptation prompt."""
    if not trades:
        return "No trades executed yet."
    
    lines = []
    for i, trade in enumerate(trades[-15:], 1):  # Last 15 trades
        action = trade.get('action', 'HOLD')
        shares = trade.get('shares', 0)
        price = trade.get('price', 0)
        tick = trade.get('tick', 0)
        pnl = trade.get('pnl', 0)
        
        pnl_str = f"(P&L: ${pnl:+.2f})" if pnl != 0 else ""
        lines.append(f"  {i}. Tick {tick}: {action} {shares} shares @ ${price:.2f} {pnl_str}")
    
    return "\n".join(lines)


def _format_price_history_analysis(prices: list) -> str:
    """Format recent price action with analysis."""
    if not prices or len(prices) < 2:
        return "Insufficient price data."
    
    current = prices[-1]
    start = prices[0]
    high = max(prices)
    low = min(prices)
    
    # Calculate momentum
    if len(prices) >= 20:
        recent_20 = prices[-20:]
        momentum = (recent_20[-1] - recent_20[0]) / recent_20[0] * 100
    else:
        momentum = (current - start) / start * 100
    
    # Detect pattern
    if len(prices) >= 10:
        last_10 = prices[-10:]
        ups = sum(1 for i in range(1, len(last_10)) if last_10[i] > last_10[i-1])
        if ups >= 7:
            pattern = "Strong Uptrend 📈"
        elif ups >= 5:
            pattern = "Mild Uptrend"
        elif ups <= 3:
            pattern = "Downtrend 📉"
        else:
            pattern = "Choppy/Sideways ↔️"
    else:
        pattern = "Unknown"
    
    return f"""  - Session Start: ${start:.2f}
  - Current Price: ${current:.2f}
  - Session High: ${high:.2f}
  - Session Low: ${low:.2f}
  - Price Range: ${high - low:.2f} ({(high - low) / start * 100:.2f}%)
  - Momentum (20-bar): {momentum:+.2f}%
  - Recent Pattern: {pattern}"""


def _calculate_trade_statistics(trades: list) -> dict:
    """Calculate detailed trade statistics."""
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'profit_factor': 0,
            'buy_count': 0,
            'sell_count': 0,
            'net_direction': 'Balanced',
        }
    
    wins = [t for t in trades if t.get('pnl', 0) > 0]
    losses = [t for t in trades if t.get('pnl', 0) < 0]
    buys = [t for t in trades if t.get('action') == 'BUY']
    sells = [t for t in trades if t.get('action') == 'SELL']
    
    total_wins = sum(t.get('pnl', 0) for t in wins)
    total_losses = abs(sum(t.get('pnl', 0) for t in losses))
    
    return {
        'total_trades': len(trades),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'avg_win': total_wins / len(wins) if wins else 0,
        'avg_loss': total_losses / len(losses) if losses else 0,
        'largest_win': max((t.get('pnl', 0) for t in wins), default=0),
        'largest_loss': min((t.get('pnl', 0) for t in losses), default=0),
        'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0,
        'buy_count': len(buys),
        'sell_count': len(sells),
        'net_direction': 'LONG-biased' if len(buys) > len(sells) * 1.2 else 'SHORT-biased' if len(sells) > len(buys) * 1.2 else 'Balanced',
    }


def _identify_algorithm_problems(roi: float, trade_stats: dict, shares_held: int, trend: str) -> list:
    """Identify specific problems with the algorithm's performance."""
    problems = []
    
    win_rate = trade_stats.get('win_rate', 0)
    total_trades = trade_stats.get('total_trades', 0)
    profit_factor = trade_stats.get('profit_factor', 1)
    avg_win = trade_stats.get('avg_win', 0)
    avg_loss = trade_stats.get('avg_loss', 0)
    net_direction = trade_stats.get('net_direction', 'Balanced')
    
    # ROI problems
    if roi < -5:
        problems.append("🚨 CRITICAL: Losing more than 5% - check stop-losses and reduce position size")
    elif roi < -2:
        problems.append("⚠️ WARNING: Losing money - tighten stop-losses, consider reducing trades")
    elif roi < 0:
        problems.append("📉 MINOR: Slightly negative ROI - fine-tuning entry/exit timing needed")
    
    # Position size problems - TARGET: 100-150 shares
    if abs(shares_held) > 200:
        problems.append(f"⚠️ OVER-POSITIONED: {abs(shares_held)} shares is too many - cap at 150")
    elif abs(shares_held) < 50 and total_trades > 5:
        problems.append(f"🚨 UNDER-POSITIONED: Only {abs(shares_held)} shares - build to 100-150 shares!")
    elif abs(shares_held) < 100 and total_trades > 10:
        problems.append(f"📦 NEED MORE: {abs(shares_held)} shares - target 100-150 shares")
    
    # Win rate problems
    if win_rate < 30 and total_trades > 5:
        problems.append(f"❌ LOW WIN RATE: {win_rate:.0f}% - entry signals are poor, be more selective")
    elif win_rate < 45 and total_trades > 10:
        problems.append(f"⚠️ BELOW AVERAGE WIN RATE: {win_rate:.0f}% - consider tightening entry criteria")
    
    # Trade frequency problems - ONLY 5-10 TOTAL TRADES is optimal
    if total_trades > 50:
        problems.append("🔄 SEVERE OVER-TRADING: 50+ trades = death by transaction costs! Target: 5 TOTAL trades MAX!")
    elif total_trades > 10:
        problems.append("🔄 OVER-TRADING: Too many trades - target ONLY 5 total trades (3 buys, 2 sells)")
    elif total_trades < 3:
        problems.append("😴 UNDER-TRADING: Build 250-share position NOW - need 3 buys in first 10 ticks!")
    
    # Risk/reward problems
    if avg_loss > avg_win * 1.5 and total_trades > 5:
        problems.append(f"⚖️ BAD RISK/REWARD: Avg loss ${avg_loss:.2f} > avg win ${avg_win:.2f} - implement stop-losses")
    
    if profit_factor < 0.8 and total_trades > 5:
        problems.append(f"📊 POOR PROFIT FACTOR: {profit_factor:.2f} - losses outweigh wins, cut losers faster")
    
    # Trend alignment
    if trend == 'BULLISH' and net_direction == 'SHORT-biased':
        problems.append("🔀 TREND MISMATCH: Shorting in a bullish market - consider reversing to LONG")
    elif trend == 'BEARISH' and net_direction == 'LONG-biased':
        problems.append("🔀 TREND MISMATCH: Going long in a bearish market - consider shorting")
    
    if not problems:
        problems.append("✅ NO MAJOR ISSUES: Algorithm is performing reasonably well")
    
    return problems


def build_adaptation_prompt(
    ticker: str,
    current_algo_code: str,
    current_roi: float,
    current_cash: float,
    current_shares: int,
    current_tick: int,
    total_ticks: int,
    price_history: List[float],
    checkpoint_num: int,
    trades: List[Dict] = None,
    total_checkpoints: int = 5
) -> str:
    """
    Build a comprehensive prompt for mid-simulation algorithm adaptation.
    
    This prompt is given to the AI agent when the simulation pauses at checkpoints
    to allow the agent to analyze its performance and potentially improve its algorithm.
    
    NOW INCLUDES:
    - Full algorithm code for review
    - Detailed trade statistics
    - Identified problems with specific fixes
    - Price action analysis
    - Trade history
    
    Args:
        ticker: Stock symbol being traded
        current_algo_code: The current algorithm code
        current_roi: Current ROI as a decimal (e.g., -0.15 for -15%)
        current_cash: Current cash balance
        current_shares: Current shares held (can be negative for short positions)
        current_tick: Current tick number in simulation
        total_ticks: Total ticks in simulation (390)
        price_history: Recent price history (last 50-100 prices)
        checkpoint_num: Which checkpoint (1-5)
        trades: List of trade dicts with 'action', 'shares', 'price', 'tick', 'pnl'
        total_checkpoints: Total number of checkpoints (default 5)
    
    Returns:
        Prompt string for algorithm adaptation
    """
    trades = trades or []
    
    roi_pct = current_roi * 100
    ticks_remaining = total_ticks - current_tick
    time_remaining_pct = ticks_remaining / total_ticks * 100
    
    # Calculate detailed trade statistics
    trade_stats = _calculate_trade_statistics(trades)
    
    # Calculate current portfolio value
    current_price = price_history[-1] if price_history else 0
    total_value = current_cash + (current_shares * current_price)
    unrealized_pnl = current_shares * (current_price - (trades[-1].get('price', current_price) if trades else current_price)) if current_shares != 0 else 0
    
    # Determine trend from price history
    trend = "UNKNOWN"
    if len(price_history) >= 20:
        if price_history[-1] > price_history[-20]:
            trend = "BULLISH"
        elif price_history[-1] < price_history[-20]:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
    
    # Identify specific problems
    problems = _identify_algorithm_problems(roi_pct, trade_stats, current_shares, trend)
    
    # Determine urgency based on time remaining and ROI target
    roi_needed_for_30 = 30 - roi_pct
    if time_remaining_pct > 60:
        time_advice = f"You have plenty of time. Need {roi_needed_for_30:.1f}% more to reach 30% ROI. Build to 100-150 shares!"
    elif time_remaining_pct > 30:
        time_advice = f"Mid-session - {roi_needed_for_30:.1f}% needed for 30% target. Ensure you're at MAX position (100-150 shares)!"
    else:
        time_advice = f"⚠️ LATE SESSION - Only {roi_needed_for_30:.1f}% needed. Start closing positions to lock in gains."
    
    # Add short selling advice for bearish markets
    if trend == "BEARISH":
        short_advice = "\n🔴 **MARKET IS BEARISH** - Consider SHORT SELLING! Go to -100 to -150 shares."
    elif trend == "BULLISH":
        short_advice = "\n🟢 **MARKET IS BULLISH** - GO LONG! Build to 100-150 shares."
    else:
        short_advice = "\n🟡 **MARKET IS NEUTRAL** - Pick a direction and GO ALL IN!"
    
    # Format the prompt
    prompt = f'''# 🔄 MID-SIMULATION ALGORITHM ADAPTATION
## CHECKPOINT {checkpoint_num} OF {total_checkpoints} | Tick {current_tick}/{total_ticks}

You are an AI trading algorithm in a competitive simulation. The simulation has PAUSED
to give you a chance to REVIEW and IMPROVE your algorithm based on real performance data.

{'='*70}
## 📊 YOUR CURRENT ALGORITHM CODE
{'='*70}

```python
{current_algo_code}
```

{'='*70}
## 📈 PERFORMANCE METRICS
{'='*70}

### Portfolio Status:
| Metric | Value |
|--------|-------|
| **Current ROI** | **{roi_pct:+.2f}%** |
| Cash Balance | ${current_cash:,.2f} |
| Shares Held | {current_shares} |
| Unrealized P&L | ${unrealized_pnl:+,.2f} |
| Total Portfolio Value | ${total_value:,.2f} |
| Ticks Remaining | {ticks_remaining} ({time_remaining_pct:.0f}% of session) |

### Trade Statistics:
| Metric | Value |
|--------|-------|
| Total Trades | {trade_stats['total_trades']} |
| Winning Trades | {trade_stats['winning_trades']} ({trade_stats['win_rate']:.1f}%) |
| Losing Trades | {trade_stats['losing_trades']} |
| Average Win | ${trade_stats['avg_win']:.2f} |
| Average Loss | ${trade_stats['avg_loss']:.2f} |
| Largest Win | ${trade_stats['largest_win']:.2f} |
| Largest Loss | ${trade_stats['largest_loss']:.2f} |
| Profit Factor | {trade_stats['profit_factor']:.2f} |
| Trade Bias | {trade_stats['net_direction']} |

{'='*70}
## 📉 MARKET CONDITIONS
{'='*70}

{_format_price_history_analysis(price_history)}
  - Current Trend: **{trend}**

{'='*70}
## 📜 RECENT TRADE HISTORY
{'='*70}

{_format_recent_trades(trades)}

{'='*70}
## ⚠️ IDENTIFIED PROBLEMS
{'='*70}

{chr(10).join(problems)}

{'='*70}
## ⏰ TIME CONTEXT
{'='*70}

{time_advice}
{short_advice}

{'='*70}
## 🎯 YOUR TASK - ACHIEVE 10%+ ROI (AGGRESSIVE TRADING)
{'='*70}

**Current ROI: {roi_pct:+.2f}%** | Target: 10%+ | Need: {max(0, 10 - roi_pct):.1f}% more**
**Current Position: {current_shares} shares | NEED: 100-150 shares | Market Trend: {trend}**

{"🚨 CRITICAL: You only have " + str(abs(current_shares)) + " shares! BUILD TO 100-150!" if abs(current_shares) < 80 else "✅ Good position size! HOLD until tick 900!"}

## 🚀 THE FIX: BUILD 100-150 SHARE POSITION

{"### 📈 GO LONG - BUY MORE!" if trend != "BEARISH" else "### 📉 GO SHORT - SELL MORE!"}

**YOU NEED {max(0, 100 - abs(current_shares))} MORE SHARES!**

### EXACT CODE TO USE:

```python
def execute_trade(ticker, price, tick, cash_balance, shares_held):
    # BUILD TO 100-150 SHARES
    if abs(shares_held) < 100:
        return ("{"BUY" if trend != "BEARISH" else "SELL"}", min(50, 100 - abs(shares_held)))
    
    # HOLD UNTIL CLOSE
    if tick < 900:
        return "HOLD"
    
    # CLOSE POSITION
    if shares_held > 0:
        return ("SELL", shares_held)
    elif shares_held < 0:
        return ("BUY", abs(shares_held))
    return "HOLD"
```

### ⚠️ RULES:
1. **10-15 TRADES MAX** - You have {trade_stats['total_trades']} already!
2. **100-150 SHARES TARGET** - You have {abs(current_shares)}!
3. **HOLD AFTER POSITION BUILT** - Let profits run!
```

### KEY INSIGHT:
- With {current_shares} shares, a 5% move = ${abs(current_shares) * float(price_history[-1]) * 0.05:.0f} profit
- With {'100' if trend != 'BEARISH' else '-100'} shares, same 5% move = ${100 * float(price_history[-1]) * 0.05:.0f} profit
- With {'150' if trend != 'BEARISH' else '-150'} shares, same 5% move = ${150 * float(price_history[-1]) * 0.05:.0f} profit

**{'GO LONG!' if trend != 'BEARISH' else 'GO SHORT!'} BUILD BIGGER POSITION!**

{'='*70}
## 🏆 EXTREME AGGRESSION EXAMPLE
{'='*70}

```python
_prices = []

def execute_trade(ticker, price, tick, cash_balance, shares_held):
    global _prices
    _prices.append(price)
    
    # DETECT TREND IN FIRST 15 TICKS
    if tick <= 15:
        return "HOLD"  # Observe market direction
    
    # DETERMINE DIRECTION
    if tick == 16:
        if _prices[-1] > _prices[0]:  # BULLISH
            return ("BUY", 100)  # GO LONG
        else:  # BEARISH
            return ("SELL", 100)  # GO SHORT
    
    # BUILD BIGGER POSITION (ticks 17-50)
    if tick <= 50:
        if shares_held > 0 and shares_held < 150:
            return ("BUY", 25)  # Add to long
        elif shares_held < 0 and shares_held > -150:
            return ("SELL", 25)  # Add to short
    
    # HOLD THROUGH EVERYTHING (ticks 50-900)
    if tick < 900:
        return "HOLD"
    
    # CLOSE POSITION AT END
    if shares_held > 0:
        return ("SELL", shares_held)  # Close long
    elif shares_held < 0:
        return ("BUY", abs(shares_held))  # Cover short
    
    return "HOLD"
```

{'='*70}
## ⚡ CRITICAL RULES FOR EXTREME AGGRESSION
{'='*70}

1. **FUNCTION SIGNATURE**: `def execute_trade(ticker, price, tick, cash_balance, shares_held):`
2. **RETURN VALUES**: `"BUY"`, `"SELL"`, `"HOLD"`, or `("BUY", N)`, `("SELL", N)` for specific quantities
3. **GO BIG BOTH WAYS**: 
   - BULLISH: +100 to +150 shares (LONG)
   - BEARISH: -100 to -150 shares (SHORT)
4. **HOLD PHASE**: Don't trade between ticks 50-900
5. **MARGIN IS OK**: Cash can go negative - use leverage!
6. **SHORT SELLING**: Shares can be negative! SELL when you have 0 shares = SHORT
7. **BIDIRECTIONAL**: Make money in UP or DOWN markets!

{'='*70}
## 📤 OUTPUT FORMAT
{'='*70}

Output ONLY the improved Python code. No explanations, no markdown fences, no commentary.
Start directly with variable declarations or `def execute_trade`.
The code must define `execute_trade` with the EXACT signature shown above.

**CURRENT TREND: {trend}**
{"GO LONG 100-150 shares!" if trend == "BULLISH" else "GO SHORT -100 to -150 shares!" if trend == "BEARISH" else "Pick direction and GO ALL IN!"}

Your improved algorithm:'''

    return prompt


async def regenerate_algorithm_for_adaptation_async(
    client: httpx.AsyncClient,
    model_id: str,
    adaptation_prompt: str,
    max_retries: int = 2
) -> Tuple[bool, Optional[str]]:
    """
    Regenerate an algorithm during mid-simulation adaptation.
    
    Returns:
        Tuple of (should_update: bool, new_code: Optional[str])
        - If agent says KEEP_ALGORITHM: (False, None)
        - If agent provides new code: (True, new_code)
        - If error: (False, None)
    """
    print(f"\n--- Adaptation request for: {model_id} ---")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title": OPENROUTER_APP_NAME,
    }
    
    for attempt in range(max_retries + 1):
        data = {"model": model_id, "messages": [{"role": "user", "content": adaptation_prompt}]}
        
        try:
            response = await client.post(
                CHAT_API_URL,
                headers=headers,
                json=data,
                timeout=45.0  # Shorter timeout for adaptation
            )

            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < max_retries:
                    delay = 1.0 * (2 ** attempt)
                    print(f"⚠️ {model_id}: HTTP {response.status_code}, retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(f"❌ Adaptation failed after retries for {model_id}")
                    return (False, None)

            response.raise_for_status()
            raw = response.json()['choices'][0]['message']['content']
            content = raw.strip()
            
            # Check if agent wants to keep current algorithm
            if 'KEEP_ALGORITHM' in content.upper():
                print(f"✅ {model_id} chose to KEEP current algorithm")
                return (False, None)
            
            # Try to extract new code
            extracted_code = _extract_execute_trade_code(content)
            
            if extracted_code and 'def execute_trade' in extracted_code and _validate_python_syntax(extracted_code):
                print(f"✅ {model_id} provided NEW algorithm for adaptation")
                return (True, extracted_code.strip())
            else:
                print(f"⚠️ {model_id} response unclear or invalid syntax, keeping current algorithm")
                return (False, None)

        except httpx.TimeoutException:
            if attempt < max_retries:
                print(f"⚠️ {model_id}: Timeout, retrying...")
                await asyncio.sleep(1.0)
                continue
            print(f"❌ Adaptation timeout for {model_id}")
            return (False, None)

        except Exception as e:
            print(f"❌ Adaptation error for {model_id}: {e}")
            return (False, None)

    return (False, None)

# --- 2. Core Algorithm Generation Functions ---

async def generate_algorithm_async(
    client: httpx.AsyncClient,
    model_id: str,
    prompt_text: str,
    fallback_prompt_text: Optional[str] = None,
    max_retries: int = 3
) -> Optional[str]:
    """
    Async version: Sends the generation prompt to a specific model and returns its response.
    Implements retry with exponential backoff + jitter for 429/5xx errors.
    If fallback_prompt_text is provided, it switches to it upon timeout.
    """
    print(f"\n--- Generating algorithm with: {model_id} ---")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title": OPENROUTER_APP_NAME,
    }
    
    current_prompt = prompt_text
    
    # Exponential backoff with jitter: 0.5s, 1s, 2s base + random jitter
    for attempt in range(max_retries + 1):
        data = {"model": model_id, "messages": [{"role": "user", "content": current_prompt}]}
        
        # Use a shorter timeout (30s) for the first attempt to quickly switch to lightweight prompt
        # Subsequent attempts get more time (60s) to complete with the lighter prompt
        request_timeout = 30.0 if attempt == 0 else 60.0

        try:
            response = await client.post(
                CHAT_API_URL,
                headers=headers,
                json=data,
                timeout=request_timeout
            )

            # Retry on rate limit or server errors
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < max_retries:
                    # Exponential backoff: 0.5s * (2^attempt) + jitter
                    base_delay = 0.5 * (2 ** attempt)
                    jitter = random.uniform(0, 0.1 * base_delay)
                    delay = base_delay + jitter
                    print(f"⚠️ {model_id}: HTTP {response.status_code}, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    err_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    print(f"❌ FAILED after {max_retries} retries for {model_id}. Error: {err_msg}")
                    return None

            response.raise_for_status()
            raw = response.json()['choices'][0]['message']['content']
            content = _extract_execute_trade_code(raw)

            if content and 'def execute_trade' in content and _validate_python_syntax(content):
                print(f"✅ SUCCESS: Code received from {model_id}.")
                return content.strip()
            else:
                # Extraction failed - retry if we have attempts left
                if attempt < max_retries:
                    print(f"⚠️ {model_id}: Code extraction failed, retrying (attempt {attempt + 1}/{max_retries})...")
                    # Switch to lightweight prompt if available
                    if fallback_prompt_text and current_prompt != fallback_prompt_text:
                        print(f"   Switching to lightweight prompt for retry.")
                        current_prompt = fallback_prompt_text
                    await asyncio.sleep(0.5)
                    continue
                print(f"❌ FAILED to find valid execute_trade in {model_id} output after {max_retries} attempts.")
                # Debug: show what we got
                if raw:
                    print(f"   Raw output preview: {raw[:200]}...")
                return None

        except httpx.TimeoutException as e:
            if attempt < max_retries:
                # Switch to fallback prompt if available and not already using it
                if fallback_prompt_text and current_prompt != fallback_prompt_text:
                    print(f"⚠️ {model_id}: Timeout (>{request_timeout}s). Switching to lightweight prompt for retry.")
                    current_prompt = fallback_prompt_text
                
                base_delay = 0.5 * (2 ** attempt)
                jitter = random.uniform(0, 0.1 * base_delay)
                delay = base_delay + jitter
                print(f"⚠️ {model_id}: Timeout, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
                continue
            print(f"❌ FAILED to get code from {model_id}. Error: {e}")
            return None

        except httpx.HTTPStatusError as e:
            print(f"❌ FAILED to get code from {model_id}. HTTP error: {e}")
            return None

        except (KeyError, IndexError) as e:
            print(f"❌ FAILED to parse response from {model_id}: {e}")
            return None

        except Exception as e:
            if attempt < max_retries:
                base_delay = 0.5 * (2 ** attempt)
                jitter = random.uniform(0, 0.1 * base_delay)
                delay = base_delay + jitter
                print(f"⚠️ {model_id}: Error {e}, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
                continue
            print(f"❌ FAILED to get code from {model_id}. Error: {e}")
            return None

    return None


def generate_algorithm(model_id, prompt_text: str):
    """Sends the generation prompt to a specific model and returns its response."""
    print(f"\n--- Generating algorithm with: {model_id} ---")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        # Recommended by OpenRouter to help with access/rate limits attribution
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title": OPENROUTER_APP_NAME,
    }
    data = {"model": model_id, "messages": [{"role": "user", "content": prompt_text}]}
    # Retries for transient errors (429/5xx/timeouts) and extraction failures
    backoffs = [1.0, 2.0, 3.0]
    last_err = None
    for attempt in range(len(backoffs) + 1):
        try:
            response = requests.post(CHAT_API_URL, headers=headers, json=data, timeout=180)
            # If server returns explicit error codes, decide whether to retry
            if response.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {response.status_code}: {response.text[:200]}"
                if attempt < len(backoffs):
                    time.sleep(backoffs[attempt])
                    continue
                else:
                    print(f"❌ FAILED after retries for {model_id}. Error: {last_err}")
                    return None
            response.raise_for_status()
            raw = response.json()['choices'][0]['message']['content']
            content = _extract_execute_trade_code(raw)
            if content and 'def execute_trade' in content and _validate_python_syntax(content):
                print(f"✅ SUCCESS: Code received from {model_id}.")
                return content.strip()
            else:
                # Extraction failed - retry if we have attempts left
                if attempt < len(backoffs):
                    print(f"⚠️ {model_id}: Code extraction failed, retrying (attempt {attempt + 1})...")
                    time.sleep(backoffs[attempt])
                    continue
                print(f"❌ FAILED to find valid execute_trade in {model_id} output after retries.")
                if raw:
                    print(f"   Raw output preview: {raw[:200]}...")
                return None
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            if attempt < len(backoffs):
                time.sleep(backoffs[attempt])
                continue
            err_txt = ""
            try:
                err_txt = f" | server: {response.text[:300]}" if 'response' in locals() and hasattr(response, 'text') else ""
            except Exception:
                err_txt = ""
            print(f"❌ FAILED to get code from {model_id}. Error: {e}{err_txt}")
            return None
        except (KeyError, IndexError) as e:
            print(f"❌ FAILED to parse response from {model_id}: {e}")
            return None

def save_algorithm_to_file(code, model_name):
    """Saves the generated code to a uniquely named file."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Sanitize the model name to create a valid filename (e.g., replace '/' with '_')
        safe_filename = model_name.replace('/', '_')
        output_path = os.path.join(OUTPUT_DIR, f'generated_algo_{safe_filename}.py')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"✅ Algorithm successfully saved to: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"❌ FAILED to save algorithm for {model_name}.\n   Error: {e}")

# --- 3. Main Execution ---

def main():
    """Main function to orchestrate the entire process."""
    print("--- Starting Algorithm Generation Process ---")
    # Step 1: Run the model fetching and testing process to get the list of models.
    generator_models = get_models_to_use()
    
    # Step 2: Check if a list of models was successfully returned.
    if not generator_models:
        print("\nHalting execution as no models were selected from the testing phase.")
        return None
        
    print(f"\n--- Starting Generation for {len(generator_models)} Models ---")

    # Step 2b: Ask the user which stock ticker to use
    ticker = input("Enter stock ticker (e.g. AAPL): ").strip().upper()
    if not ticker:
        ticker = "AAPL"
        print("⚠️ No ticker entered; defaulting to AAPL.")

    print(f"📌 Using ticker: {ticker} for prompt context")

    base_prompt, compressed_data = build_generation_prompt(ticker)
    
    # Step 3: Loop through each selected model, generate, and save.
    for model in generator_models:
        per_model_prompt = base_prompt + build_diversity_directives(model)
        generated_code = generate_algorithm(model, per_model_prompt)
        if generated_code:
            generated_code = _prepend_historical_data(generated_code, compressed_data)
            save_algorithm_to_file(generated_code, model)
        # Add a small delay to be respectful to the API
        time.sleep(2)
    
    print("\n--- All generation tasks completed. ---")
    return ticker


if __name__ == "__main__":
    main()

# --- 4. OpenRouter model availability helpers ---

def _normalize_model_id_set(model_ids: set[str]) -> set[str]:
    """Return a set including both with/without ':free' variants for match convenience."""
    out = set()
    for mid in model_ids or []:
        out.add(mid)
        if mid.endswith(":free"):
            out.add(mid[:-5])
        else:
            out.add(f"{mid}:free")
    return out

_CACHED_ACCESSIBLE: tuple[float, set[str]] | None = None

def _get_accessible_models() -> set[str] | None:
    """Query OpenRouter for models accessible to this API key. Cached for 60s."""
    global _CACHED_ACCESSIBLE
    now = time.time()
    if _CACHED_ACCESSIBLE and (now - _CACHED_ACCESSIBLE[0] < 60):
        return _CACHED_ACCESSIBLE[1]
    if not API_KEY:
        return None
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_APP_NAME,
        }
        r = requests.get(MODELS_API_URL, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        ids = set()
        for m in data.get("data", []):
            mid = m.get("id")
            if isinstance(mid, str):
                ids.add(mid)
        _CACHED_ACCESSIBLE = (now, ids)
        return ids
    except Exception as e:
        print(f"⚠️ Could not fetch accessible models list: {e}")
        return None

def _pick_best_accessible_id(selected_id: str, normalized_accessible: set[str]) -> str | None:
    """Return an accessible variant of selected_id if available; else None if list is non-empty."""
    if not normalized_accessible:
        # If we couldn't fetch, don't filter
        return selected_id
    candidates = [selected_id]
    # Try toggling ':free'
    if selected_id.endswith(":free"):
        candidates.append(selected_id[:-5])
    else:
        candidates.append(f"{selected_id}:free")
    for cand in candidates:
        if cand in normalized_accessible:
            # Return the exact ID as present in accessible set if possible
            # Normalize to one of the variants contained within the set
            if cand in normalized_accessible:
                return cand
    return None

def _validate_python_syntax(code: str) -> bool:
    """Validate that the provided code is syntactically correct Python."""
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False


def _extract_execute_trade_code(raw_content: str) -> str | None:
    """Attempt to extract a clean Python function containing execute_trade from model output.
    Handles content with markdown fences, extra prose, or multiple blocks.
    Also fixes common formatting issues like broken function signatures.
    """
    if not isinstance(raw_content, str):
        return None
    text = raw_content.strip()
    
    # Pre-process: fix broken function signatures (newline before colon)
    # This handles cases like "def execute_trade(...)\n:" -> "def execute_trade(...):"
    import re
    
    # Fix pattern: closing paren followed by newline(s) and colon
    text = re.sub(r'\)\s*\n\s*:', '):', text)
    
    # Also fix any double colons that might result
    text = text.replace('::', ':')

    # 1) If there are fenced code blocks, search for one containing the function
    blocks = []
    try:
        # Split by triple backticks while preserving content
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            lang_and_code = parts[i]
            # Remove leading language tag like 'python\n'
            if "\n" in lang_and_code:
                first, rest = lang_and_code.split("\n", 1)
                code = rest
            else:
                code = lang_and_code
            # Apply the same fix to extracted blocks
            code = re.sub(r'\)\s*\n\s*:', '):', code)
            code = code.replace('::', ':')
            blocks.append(code)
    except Exception:
        blocks = []

    # Prefer a block with execute_trade
    for b in blocks:
        if 'def execute_trade' in b:
            return b.strip()

    # 2) If no blocks or no match, try to extract from raw text by finding the function definition
    m = re.search(r"(^|\n)def\s+execute_trade\s*\(.*?\):", text)
    if m:
        start = m.start(0) if m.start(0) >= 0 else 0
        snippet = text[start:]
        # crude but effective: stop at next unindented def/class or end
        lines = snippet.splitlines()
        out = []
        for ln in lines:
            out.append(ln)
            if re.match(r"^[^\s]", ln) and ln.startswith(('def ', 'class ')) and len(out) > 1:
                break
        return "\n".join(out).strip()

    # 3) Fallback: if text itself is code and includes imports but forgot the function name, return text
    return text if 'def execute_trade' in text else None