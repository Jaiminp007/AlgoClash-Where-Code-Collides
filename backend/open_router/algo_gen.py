import os
import requests
import json
from dotenv import load_dotenv
import time
import hashlib
import random
import asyncio
import httpx
from typing import Optional, List, Tuple

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

# Data directory for local CSVs
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

def list_available_stocks(data_dir: str) -> list:
    """List available stock CSVs (filters *_data.csv) in the data directory."""
    try:
        files = [f for f in os.listdir(data_dir) if f.lower().endswith('_data.csv')]
        files.sort()
        return files
    except Exception:
        return []

 
    
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

    # Load dataset preview for prompt context
    csv_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    csv_preview = load_csv_preview(csv_path) if os.path.exists(csv_path) else ""
    base_prompt = build_generation_prompt(ticker, csv_preview)

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
                code = await generate_algorithm_async(client, agent_model, per_model_prompt)

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

def select_stock_file() -> tuple:
    """Interactively ask the user to pick a stock CSV. Returns (ticker, filename, full_path)."""
    files = list_available_stocks(DATA_DIR)
    fallback_file = 'stock_data.csv'
    if not files:
        # Fallback to stock_data.csv if present
        if os.path.exists(os.path.join(DATA_DIR, fallback_file)):
            files = [fallback_file]
        else:
            print("❌ No stock CSVs found in data directory.")
            return None, None, None

    print("\n📂 Available stock datasets (backend/data/):")
    for idx, fname in enumerate(files, 1):
        print(f"  {idx}. {fname}")

    choice = input(f"Select a dataset by number (1-{len(files)}) or press Enter for 1: ").strip()

    selected = None
    if not choice:
        selected = files[0]
    else:
        try:
            idx = int(choice)
            if 1 <= idx <= len(files):
                selected = files[idx - 1]
        except ValueError:
            pass

    # Try matching by ticker symbol if numeric selection failed
    if selected is None:
        ticker_guess = choice.upper().replace('.CSV', '').replace('_DATA', '')
        match = next((f for f in files if f.upper().startswith(f"{ticker_guess}_")), None)
        if match:
            selected = match
        else:
            print("⚠️ Invalid selection. Defaulting to 1.")
            selected = files[0]

    ticker = selected.split('_')[0].upper()
    path = os.path.join(DATA_DIR, selected)
    return ticker, selected, path

def load_csv_preview(csv_path: str, max_rows: int = 200) -> str:
    """Return header + last max_rows of CSV to keep prompt size reasonable."""
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            return ""
        header = lines[0].strip()
        data_lines = [ln.strip() for ln in lines[1:] if ln.strip()]
        preview = data_lines[-max_rows:] if len(data_lines) > max_rows else data_lines
        out = [header] + preview
        return "\n".join(out)
    except Exception:
        return ""

def build_generation_prompt(ticker: str, csv_preview: str) -> str:
    """Build a prompt that encourages original algorithm design without prescriptive templates."""
    base = f"""You are an expert quantitative trading researcher tasked with designing a unique, production-ready trading algorithm.

═══════════════════════════════════════════════════════════════════════════════
OBJECTIVE
═══════════════════════════════════════════════════════════════════════════════

Design and implement a Python function that analyzes real market data and makes intelligent trading decisions.

Your algorithm will compete against other AI-generated strategies in a live market simulation. Originality and robustness are critical.

═══════════════════════════════════════════════════════════════════════════════
FUNCTION CONTRACT (MANDATORY)
═══════════════════════════════════════════════════════════════════════════════

Function signature:
    def execute_trade(ticker: str, cash_balance: float, shares_held: int) -> str

Parameters:
    - ticker: Stock symbol (e.g., "AAPL", "TSLA")
    - cash_balance: Available cash in USD (may be negative if borrowing)
    - shares_held: Current stock position (positive for long, negative for short)

Return value:
    Must return EXACTLY one of these strings: "BUY", "SELL", or "HOLD"
    - Uppercase only, no quotes in the return statement
    - No explanations, no additional text

CRITICAL TRADING RULES:
    - BUY when you predict price will INCREASE (go long or cover shorts)
    - SELL when you predict price will DECREASE (take profit or short sell)
    - Short selling is enabled: You can SELL even with shares_held <= 0
    - PROFIT IN ANY MARKET: Make money whether prices go up OR down
    - Detect market direction and trade accordingly (long in uptrends, short in downtrends)

Output format:
    - Raw Python code ONLY
    - No markdown code fences (no ```)
    - No explanatory comments or docstrings
    - No print statements or debug output
    - Code must be immediately executable

═══════════════════════════════════════════════════════════════════════════════
EXECUTION ENVIRONMENT & DATA ACCESS
═══════════════════════════════════════════════════════════════════════════════

IMPORTANT: Your function will be called repeatedly during a live market simulation.
You will ONLY receive these parameters:
    - ticker: The stock symbol being traded
    - cash_balance: Your current cash position (can be negative with margin)
    - shares_held: Your current stock position (can be negative with short selling)

CRITICAL CONSTRAINTS:
    - You do NOT receive current market price as a parameter
    - You do NOT have access to live price feeds during execution
    - The simulation engine handles all market data internally
    - Your function will be called 60+ times during a 15-second simulation

RECOMMENDED APPROACH - Stateful Trading Logic:

1. Use module-level variables to track state across calls:
   - Example: _call_count = 0, _last_position = 0, _trades_made = 0
   - Track your trading history and position changes
   - Implement time-based or position-based strategies

2. Make decisions based on your portfolio state:
   - If shares_held > X and profitable, consider taking profit (SELL)
   - If shares_held < Y, consider building position (BUY)
   - Use cash_balance to gauge your buying power
   - Track position changes to infer market movement

3. Pattern-based strategies without price data:
   - Cycle through BUY/HOLD/SELL based on internal counters
   - Use probabilistic/random elements (with seed for reproducibility)
   - Implement mean-reversion via position sizing
   - Use portfolio rotation strategies

OPTIONAL: If you MUST use market data (NOT recommended):
   - Use yfinance with AGGRESSIVE caching: yf.download(ticker, period="5d", interval="1m", progress=False)
   - Store in module-level variable: _cached_data =
   - Check cache FIRST before downloading
   - WARNING: This adds latency and is discouraged for this simulation

Example of good stateful approach:
    _trade_cycle = 0
    _position_target = 0

    def execute_trade(ticker, cash_balance, shares_held):
        global _trade_cycle, _position_target
        _trade_cycle += 1

        # Cycle through positions
        if _trade_cycle % 20 == 0:
            _position_target = 10 if shares_held < 5 else -5

        if shares_held < _position_target:
            return "BUY"
        elif shares_held > _position_target:
            return "SELL"
        return "HOLD"

═══════════════════════════════════════════════════════════════════════════════
ERROR HANDLING (CRITICAL)
═══════════════════════════════════════════════════════════════════════════════

Your function MUST handle these edge cases gracefully:

1. Insufficient data:
   - Check if dataframe is empty or None
   - Verify array length before indexing: if len(close_prices) < required_window
   - Return "HOLD" when data is insufficient

2. NaN/Inf values:
   - Check for NaN: if np.isnan(value) or not np.isfinite(value)
   - Return "HOLD" if any critical calculation yields NaN or Inf

3. Division by zero:
   - Check denominators: if denominator <= 0 or denominator == 0
   - Return "HOLD" to avoid exceptions

4. Array operations:
   - Ensure window sizes don't exceed array length
   - Validate indices before slicing

5. Exception handling:
   - Wrap main logic in try/except
   - Return "HOLD" on any exception

═══════════════════════════════════════════════════════════════════════════════
ALGORITHM DESIGN PHILOSOPHY
═══════════════════════════════════════════════════════════════════════════════

CREATE AN ORIGINAL STRATEGY. Do not implement generic moving average crossovers or basic RSI thresholds.

PROFIT IN ANY MARKET CONDITION:
Your algorithm must be able to profit whether the market goes UP or DOWN:
    - Uptrend detected → BUY (go long) to profit from rising prices
    - Downtrend detected → SELL (go short) to profit from falling prices
    - Sideways/uncertain → HOLD or trade ranges
    - You can short sell (SELL with shares_held <= 0) to profit from declines
    - Focus on DIRECTIONAL PREDICTION, not just momentum following

Design your algorithm around a clear market hypothesis:

1. What market inefficiency or pattern are you exploiting?
2. What signals validate your hypothesis?
3. Under what specific conditions do you BUY vs SELL vs HOLD?
4. How do you filter out noise and false signals?
5. How do you detect market direction (up, down, sideways)?

Strategy approaches to consider (choose and innovate):
   - Mean reversion: Price deviations from equilibrium, Bollinger Bands, z-scores
   - Momentum: Trend following, breakouts, rate of change, MACD
   - Volatility: ATR regimes, volatility compression/expansion, Keltner Channels
   - Volume analysis: Volume spikes, OBV, volume-weighted metrics
   - Multi-factor: Combine uncorrelated signals with weighted scoring
   - Statistical: Cointegration, correlation patterns, statistical arbitrage
   - Pattern recognition: Support/resistance, candlestick patterns, chart formations
   - Adaptive: Dynamic thresholds based on market conditions
   - Risk-adjusted: Sharpe optimization, drawdown protection, volatility scaling

Technical indicators library (use as needed):
   - Moving averages: SMA, EMA, WMA
   - Oscillators: RSI, Stochastic, Williams %R, CCI
   - Trend: MACD, ADX, Aroon, Parabolic SAR
   - Volatility: ATR, Bollinger Bands, Standard Deviation
   - Volume: OBV, VWAP, Volume Rate of Change
   - Custom: Design your own derived metrics

CRITICAL: You must CALCULATE any technical indicators you want to use. Example calculations:

   RSI calculation:
       deltas = np.diff(close_prices)
       ups = np.maximum(deltas, 0)
       downs = np.abs(np.minimum(deltas, 0))
       avg_gain = np.mean(ups[-14:])
       avg_loss = np.mean(downs[-14:])
       rs = avg_gain / avg_loss if avg_loss > 0 else 0
       rsi = 100 - (100 / (1 + rs))

   SMA calculation:
       sma_20 = np.mean(close_prices[-20:])

   EMA calculation:
       alpha = 2 / (period + 1)
       ema = close_prices[0]
       for price in close_prices[1:]:
           ema = alpha * price + (1 - alpha) * ema

   Bollinger Bands:
       sma = np.mean(close_prices[-20:])
       std = np.std(close_prices[-20:])
       upper_band = sma + (2 * std)
       lower_band = sma - (2 * std)

   ATR (Average True Range):
       high = df['High'].values.flatten()
       low = df['Low'].values.flatten()
       close = df['Close'].values.flatten()
       tr = np.maximum(high[1:] - low[1:],
                      np.abs(high[1:] - close[:-1]),
                      np.abs(low[1:] - close[:-1]))
       atr = np.mean(tr[-14:])

   NEVER assume indicators are pre-calculated in the dataframe

Performance targets:
   - Aim for 30-50% actionable decisions (BUY or SELL)
   - Avoid algorithms that always return HOLD
   - Balance signal frequency with signal quality
   - CRITICAL: Profit in both UP and DOWN markets via directional trading
   - Use BUY for uptrends, SELL for downtrends (short selling enabled)
   - Your goal is ABSOLUTE RETURNS regardless of market direction

═══════════════════════════════════════════════════════════════════════════════
IMPLEMENTATION REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════

1. Imports:
   - Required: import yfinance as yf, import numpy as np
   - Optional: import pandas as pd (minimize usage for performance)
   - Do NOT import: matplotlib, sklearn, tensorflow, or external libraries

2. Structure:
   - Define cache dictionary at module level
   - Implement execute_trade function
   - All logic inside the function or helper functions

3. Variables and naming:
   - Use descriptive variable names
   - Keep code clean and readable
   - Avoid single-letter variables except in loops

4. Performance:
   - Minimize computational complexity
   - Avoid nested loops over large datasets
   - Use numpy vectorized operations when possible

5. Determinism:
   - No random number generation
   - No external API calls except yfinance
   - Algorithm should be reproducible

═══════════════════════════════════════════════════════════════════════════════
YOUR SPECIFIC ASSIGNMENT
═══════════════════════════════════════════════════════════════════════════════

Ticker: {ticker}
"""

    if csv_preview:
        preview_lines = csv_preview.split('\n')[:50]  # Limit preview size
        preview_sample = '\n'.join(preview_lines)
        base += f"""
Recent market data sample:
```
{preview_sample}
```

This is a reference. Your algorithm must fetch fresh data using yfinance.
"""

    base += """
═══════════════════════════════════════════════════════════════════════════════
FINAL INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════════

You will receive a STRATEGY DIRECTIVE below that specifies your unique trading philosophy and approach. Design your algorithm to embody that specific strategy.

Requirements checklist:
✓ Function named execute_trade with exact signature
✓ Returns "BUY", "SELL", or "HOLD" (uppercase strings)
✓ Uses yfinance with progress=False
✓ Implements caching for data downloads
✓ Handles all error cases (insufficient data, NaN, division by zero)
✓ Raw Python code only (no markdown, no comments)
✓ Original strategy design (not a generic template)

Now design your algorithm. Be creative, rigorous, and compete to win.
"""
    return base

def build_diversity_directives(model_id: str) -> str:
    """Create deterministic per-model directives focused on strategy philosophy rather than implementation."""
    # Deterministic seed from model_id
    seed_int = int(hashlib.md5(model_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_int)

    # Market hypotheses that drive different strategies
    market_hypotheses = [
        "Prices tend to revert to their mean after extreme movements - profit from reversals in BOTH directions",
        "Strong trends persist longer than random walks - ride uptrends long, downtrends short",
        "Volatility clustering creates predictable regime changes - trade directionally based on regime detection",
        "Volume precedes price - unusual volume patterns signal direction changes before they happen",
        "Multiple uncorrelated signals combined provide more reliable directional predictions",
        "Market overreactions create short-term mispricings - fade extremes by shorting peaks and buying dips",
        "Statistical anomalies in price distributions reveal systematic directional biases to exploit",
        "Price patterns repeat due to human psychology - identify formations and trade the breakout direction",
        "Momentum accelerates before reversals - detect exhaustion and trade the reversal direction",
        "Markets alternate between trending and ranging phases - go directional in trends, neutral in ranges",
        "Support and resistance levels create predictable bounces and breakouts - trade both directions",
        "Rate of change divergences signal impending direction changes - front-run the move",
        "Intraday volatility patterns are predictable - trade the dominant intraday direction",
        "Correlation breakdowns signal regime shifts - reposition directionally for the new regime",
        "Directional bias emerges from multiple timeframe alignment - trade when timeframes agree",
        "Risk-adjusted returns come from accurate directional prediction, not just momentum following",
        "Market microstructure imbalances reveal short-term directional edges to exploit",
        "Sentiment extremes mark turning points - short at euphoria, buy at panic"
    ]

    # Time horizons with philosophy rather than exact parameters
    time_horizons = [
        "ULTRA-SHORT-TERM: Focus on recent price action and intraday patterns (minutes to hours)",
        "SHORT-TERM: Capitalize on daily volatility and short-term trends (hours to days)",
        "MEDIUM-TERM: Trade swing movements and weekly patterns (days to weeks)",
        "LONG-TERM: Capture major trends and longer-cycle patterns (weeks to months)",
        "MULTI-TIMEFRAME: Synthesize signals across multiple time scales for confirmation"
    ]

    # Risk profiles that guide trading behavior
    risk_profiles = [
        "AGGRESSIVE: High frequency trading with tight entry/exit thresholds, accept more false signals",
        "CONSERVATIVE: Selective trading requiring multiple confirmations, prioritize accuracy over frequency",
        "BALANCED: Moderate frequency with reasonable filters, balance signal quality and quantity",
        "ADAPTIVE: Dynamically adjust aggression based on market volatility and recent performance",
        "CONTRARIAN: Trade against prevailing sentiment when indicators show extremes"
    ]

    # Signal combination approaches
    signal_approaches = [
        "Use a single powerful indicator with optimal parameters and strong filters",
        "Combine two complementary indicators (e.g., trend + momentum) with clear crossover logic",
        "Build a multi-factor scoring model weighting 3+ uncorrelated signals",
        "Design a custom composite metric that captures your unique market view",
        "Use regime detection to switch between different sub-strategies",
        "Implement statistical scoring based on historical distributions and z-scores",
        "Create adaptive thresholds that adjust based on recent market behavior"
    ]

    # Specific focus areas for differentiation
    focus_areas = [
        "Price action: Focus on OHLC patterns, candlestick formations, and price momentum",
        "Volume analysis: Emphasize volume patterns, volume-weighted metrics, and OBV",
        "Volatility metrics: Use ATR, standard deviation, Bollinger Bands, and volatility regimes",
        "Oscillators: Leverage RSI, Stochastic, Williams %R, or CCI for overbought/oversold",
        "Trend indicators: Employ moving averages, MACD, ADX, or custom trend metrics",
        "Statistical measures: Apply z-scores, percentile ranks, or distribution analysis",
        "Rate of change: Analyze acceleration, momentum derivatives, and velocity metrics",
        "Support/resistance: Identify and trade key price levels using historical patterns"
    ]

    # Select random elements based on model_id seed
    hypothesis = rng.choice(market_hypotheses)
    time_horizon = rng.choice(time_horizons)
    risk_profile = rng.choice(risk_profiles)
    signal_approach = rng.choice(signal_approaches)
    focus_area = rng.choice(focus_areas)

    # Create a unique strategy philosophy statement
    dir_text = f"""
═══════════════════════════════════════════════════════════════════════════════
YOUR UNIQUE STRATEGY MANDATE
═══════════════════════════════════════════════════════════════════════════════

Model ID: {model_id}

CORE MARKET HYPOTHESIS:
{hypothesis}

TIME HORIZON:
{time_horizon}

RISK PROFILE:
{risk_profile}

SIGNAL APPROACH:
{signal_approach}

TECHNICAL FOCUS:
{focus_area}

═══════════════════════════════════════════════════════════════════════════════
IMPLEMENTATION GUIDANCE
═══════════════════════════════════════════════════════════════════════════════

Design your algorithm to embody this specific philosophy. Your implementation should:

1. Choose data period and interval that align with your time horizon
2. Select indicators and metrics that validate your market hypothesis
3. Set thresholds and filters consistent with your risk profile
4. Implement signal logic that matches your chosen approach
5. Emphasize the technical focus area specified above

KEY REQUIREMENTS:
- Be distinctly different from generic template strategies
- Make the hypothesis testable through your indicator choices
- Ensure the risk profile is reflected in your trading frequency and thresholds
- Target 30-50% actionable decisions (BUY or SELL) under typical market conditions
- CRITICAL: Your algorithm MUST be able to profit in BOTH rising and falling markets
- Detect market direction and trade accordingly (long when bullish, short when bearish)
- Don't just follow momentum - predict direction and position accordingly

DIFFERENTIATION:
Your algorithm must be recognizably different from other models. Avoid:
- Generic dual moving average crossovers with standard parameters
- Basic RSI > 70 / RSI < 30 threshold strategies
- Simple Bollinger Band breakout systems without additional logic
- Strategies that always return HOLD

Instead, create sophisticated logic that:
- Combines multiple signals in novel ways
- Uses adaptive or dynamic thresholds
- Implements multi-layer filters for signal quality
- Reflects deep understanding of your market hypothesis

═══════════════════════════════════════════════════════════════════════════════

Now implement your original algorithm. Make it compete to win.
"""
    return dir_text

# --- 2. Core Algorithm Generation Functions ---

async def generate_algorithm_async(
    client: httpx.AsyncClient,
    model_id: str,
    prompt_text: str,
    max_retries: int = 3
) -> Optional[str]:
    """
    Async version: Sends the generation prompt to a specific model and returns its response.
    Implements retry with exponential backoff + jitter for 429/5xx errors.
    """
    print(f"\n--- Generating algorithm with: {model_id} ---")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title": OPENROUTER_APP_NAME,
    }
    data = {"model": model_id, "messages": [{"role": "user", "content": prompt_text}]}

    # Exponential backoff with jitter: 0.5s, 1s, 2s base + random jitter
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(
                CHAT_API_URL,
                headers=headers,
                json=data,
                timeout=60.0  # Per-request timeout
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

            if content and 'def execute_trade' in content:
                print(f"✅ SUCCESS: Code received from {model_id}.")
                return content.strip()
            else:
                print(f"❌ FAILED to find execute_trade in {model_id} output.")
                return None

        except httpx.TimeoutException as e:
            if attempt < max_retries:
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
    # Retries for transient errors (429/5xx/timeouts)
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
            if content and 'def execute_trade' in content:
                print(f"✅ SUCCESS: Code received from {model_id}.")
                return content.strip()
            else:
                print(f"❌ FAILED to find execute_trade in {model_id} output.")
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

    # Step 2b: Ask the user which stock dataset to use as context
    ticker, filename, csv_path = select_stock_file()
    if ticker and csv_path:
        csv_preview = load_csv_preview(csv_path, max_rows=200)
        print(f"📌 Using dataset: {filename} (ticker {ticker}) for prompt context")
    else:
        ticker = "AAPL"
        csv_preview = ""
        print("⚠️ Proceeding without local CSV context; defaulting ticker to AAPL for prompt.")

    base_prompt = build_generation_prompt(ticker, csv_preview)
    
    # Step 3: Loop through each selected model, generate, and save.
    for model in generator_models:
        per_model_prompt = base_prompt + build_diversity_directives(model)
        generated_code = generate_algorithm(model, per_model_prompt)
        if generated_code:
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

def _extract_execute_trade_code(raw_content: str) -> str | None:
    """Attempt to extract a clean Python function containing execute_trade from model output.
    Handles content with markdown fences, extra prose, or multiple blocks.
    """
    if not isinstance(raw_content, str):
        return None
    text = raw_content.strip()

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
            blocks.append(code)
    except Exception:
        blocks = []

    # Prefer a block with execute_trade
    for b in blocks:
        if 'def execute_trade' in b:
            return b.strip()

    # 2) If no blocks or no match, try to extract from raw text by finding the function definition
    import re
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