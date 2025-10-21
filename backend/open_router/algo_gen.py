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
from model_fecthing import get_models_to_use
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
    """Produce a safe, diversified minimal algorithm using yfinance when API is unavailable.
    Diversification is seeded by model_id to yield different windows/thresholds per agent.
    """
    seed_int = int(hashlib.md5(model_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_int)
    # Vary period/interval and windows
    period = rng.choice(['5d', '10d', '30d'])
    interval = rng.choice(['1m', '5m', '15m'])
    fast = rng.choice([5, 7, 9, 11])
    slow = rng.choice([15, 21, 27, 33, 45])
    # Ensure slow > fast
    if slow <= fast:
        slow = fast + rng.choice([8, 12, 20])
    buy_mult = 1.0 + rng.uniform(0.0003, 0.0012)  # 3-12 bps
    sell_mult = 1.0 - rng.uniform(0.0003, 0.0012)
    use_rsi = rng.choice([True, False])
    rsi_win = rng.choice([7, 10, 14, 21])
    cache_name = f"_fb_{hashlib.md5((model_id+'-cache').encode()).hexdigest()[:6]}"

    code = [
        "import yfinance as yf",
        "import numpy as np",
        f"{cache_name} = {{}}",
        "",
        "def execute_trade(ticker, cash_balance, shares_held):",
        f"    global {cache_name}",
        "    try:",
        f"        if ticker not in {cache_name}:",
        f"            {cache_name}[ticker] = yf.download(ticker, period='{period}', interval='{interval}', progress=False)",
        f"        df = {cache_name}.get(ticker)",
        "        if df is None or len(df) < 20:",
        "            return 'HOLD'",
        "        close_prices = df['Close'].values.flatten()",
        "        n = len(close_prices)",
        "        if n < max(20, %d):" % (max(fast, slow)),
        "            return 'HOLD'",
        f"        ma_fast = float(np.mean(close_prices[-{fast}:]))",
        f"        ma_slow = float(np.mean(close_prices[-{slow}:]))",
        "        if np.isnan(ma_fast) or np.isnan(ma_slow):",
        "            return 'HOLD'",
    ]

    if use_rsi:
        code += [
            f"        # RSI filter",
            f"        if n < {rsi_win}:",
            "            return 'HOLD'",
            f"        deltas = np.diff(close_prices[-({rsi_win}+1):])",
            "        ups = np.sum(deltas[deltas > 0])",
            "        downs = -np.sum(deltas[deltas < 0])",
            "        if downs <= 0:",
            "            return 'HOLD'",
            "        rs = ups / downs",
            "        rsi = 100.0 - (100.0 / (1.0 + rs))",
            "        if np.isnan(rsi):",
            "            return 'HOLD'",
        ]

    code += [
        f"        if ma_fast > ma_slow * {buy_mult:.6f}",
        "            return 'BUY'",
        f"        if ma_fast < ma_slow * {sell_mult:.6f}",
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
                    progress_callback(step_prog, f"Generating algorithm {index+1}/{total} using {agent_model}...")
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
    """Build a dynamic prompt that instructs the model and enforces high diversity across agents."""
    base = f"""
You are an expert quantitative trading researcher. Write a single Python function `execute_trade(ticker, cash_balance, shares_held)` that returns one of: "BUY", "SELL", or "HOLD".

Contract:
- Signature: `def execute_trade(ticker: str, cash_balance: float, shares_held: int) -> str`
- Return ONLY one of: BUY | SELL | HOLD (uppercase, no punctuation)
- Output MUST be only raw Python code (no markdown/comments)
- Use real market data via yfinance; no synthetic prices or randomness

Data requirements:
- `import yfinance as yf` and download data with period/interval consistent with DIRECTIVES below; always set `progress=False`
- Use `close_prices = df['Close'].values.flatten()` for arrays and check lengths before computations
- Avoid pandas-heavy ops in the decision path; prefer numpy/array operations
- Cache the dataframe in a module-level dict to avoid repeated downloads (exact cache name provided in DIRECTIVES)

Error safety (mandatory):
- Guard against empty/short arrays; check `len(close_prices)` before slicing
- Handle NaN: if any computed value is NaN, return HOLD
- Prevent division-by-zero; when denominator <= 0, return HOLD
- Ensure np.convolve or rolling calcs only run with sufficient data; otherwise HOLD

Diversity and uniqueness mandates:
- Each model MUST implement a distinct strategy profile chosen from categories below (also reinforced by DIRECTIVES per model)
- Vary indicators, lookbacks, thresholds, and decision logic; do NOT reuse the example values
- Be action-oriented: target ~30–50% BUY/SELL overall under typical conditions (not always HOLD)
- Use variable names prefixed with a unique per-model prefix (provided in DIRECTIVES) and the exact cache variable name provided

Strategy categories (choose ONE primary focus and may combine with a secondary):
- MEAN REVERSION (deviations from MA/bands)
- MOMENTUM/TREND (breakouts, crossovers, slope)
- VOLATILITY (ATR/StdDev regimes, squeeze/expansion)
- VOLUME CONFIRMATION (OBV/volume filters)
- OSCILLATORS (RSI/MACD/Stochastic/Williams %R)
- TIME REGIME (session time buckets)
- RISK-ADJUSTED (volatility scaling, drawdown guards)

Period/interval options (choose ONE):
- SHORT: 5–10 days intraday (e.g., 5d/1m, 10d/1m)
- MEDIUM: 15–45 days (e.g., 30d/15m, 45d/30m)
- LONG: 60–90 days (e.g., 60d/30m, 90d/1h)
- MIXED: compute features from multiple downloads (keep minimal and cache separately)

Indicator combination (choose ONE):
- SINGLE INDICATOR
- DUAL CROSSOVER
- MULTI-FACTOR (3+ signals combined)
- CUSTOM METRIC (design your own)

Threshold style (choose ONE):
- AGGRESSIVE (frequent trades)
- MODERATE (balanced)
- CONSERVATIVE (selective)
- ADAPTIVE (based on recent volatility)

Implementation guidance:
- Derive windows from array length or provided parameters; avoid the sample `window=10`
- Prefer prime or non-trivial window choices (e.g., 7/11/19) and parameterize with provided DIRECTIVES
- Gate signals with sanity checks (trend filter, volatility floor, volume confirm) to reduce noise
- Keep the function pure (no printing, no files, no globals except the cache dict)

Example pattern (do NOT copy values; adapt logic safely):
import yfinance as yf
import numpy as np

_unique_cache = {{}}

def execute_trade(ticker, cash_balance, shares_held):
    global _unique_cache
    if ticker not in _unique_cache:
        _unique_cache[ticker] = yf.download(ticker, period="5d", interval="1m", progress=False)
    df = _unique_cache.get(ticker)
    if df is None or len(df) < 20:
        return "HOLD"
    close_prices = df['Close'].values.flatten()
    if len(close_prices) < 20:
        return "HOLD"
    # compute indicators...
    return "HOLD"

You will ALSO receive DIRECTIVES below that specify a unique cache name, variable prefix, period/interval, and required parameter values for THIS model. You MUST follow them exactly.

Now create a UNIQUE algorithm for ticker {ticker} following the DIRECTIVES.

CRITICAL checks list:
1) Use `close_prices = df['Close'].values.flatten()`
2) Check `len(close_prices) >= needed_window` before slicing
3) Handle NaNs and division-by-zero by returning HOLD
4) Keep names ASCII-only; use the provided prefix for all new variables
5) Actionable: aim for ~30–50% BUY/SELL overall (avoid always HOLD)

Context preview (local CSV tail for calibration; still fetch with yfinance):
"""
    if csv_preview:
        base += f"\n```\n{csv_preview}\n```\n"
    return base

def build_diversity_directives(model_id: str) -> str:
    """Create deterministic per-model directives to enforce diverse strategies with concrete parameters."""
    # Deterministic seed from model_id
    seed_int = int(hashlib.md5(model_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_int)

    strategy_types = [
        "MEAN REVERSION", "MOMENTUM", "VOLATILITY", "VOLUME-BASED",
        "TECHNICAL INDICATORS", "PATTERN RECOGNITION", "TIME-BASED",
        "RISK-ADJUSTED", "ARBITRAGE", "CONTRARIAN"
    ]
    data_periods = [
        ("SHORT-TERM", "5d", "1m"), ("SHORT-TERM", "10d", "1m"),
        ("MEDIUM-TERM", "30d", "15m"), ("MEDIUM-TERM", "45d", "30m"),
        ("LONG-TERM", "60d", "30m"), ("LONG-TERM", "90d", "1h"),
        ("MIXED", "30d", "15m")
    ]
    indicator_combo = ["SINGLE INDICATOR", "DUAL CROSSOVER", "MULTI-FACTOR", "CUSTOM METRIC"]
    thresholds = ["AGGRESSIVE", "MODERATE", "CONSERVATIVE", "ADAPTIVE"]
    logic_types = ["ALWAYS BUY/SELL", "CONDITIONAL", "PROBABILISTIC", "THRESHOLD-BASED"]
    portfolio_styles = ["AGGRESSIVE", "CONSERVATIVE", "BALANCED", "ADAPTIVE"]

    strat = rng.choice(strategy_types)
    period_label, period_val, interval_val = rng.choice(data_periods)
    combo = rng.choice(indicator_combo)
    thresh = rng.choice(thresholds)
    logic = rng.choice(logic_types)
    port = rng.choice(portfolio_styles)

    # Unique variable prefix and cache name
    var_prefix = f"v{hashlib.md5((model_id+'-vars').encode()).hexdigest()[:6]}_"
    cache_name = f"_{var_prefix}cache"

    # Deterministic numeric parameters
    fast_ma = rng.choice([5, 7, 9, 11])
    slow_ma = rng.choice([20, 30, 45, 60])
    rsi_win = rng.choice([7, 10, 14, 21])
    bb_win = rng.choice([12, 18, 20, 24])
    vol_lb = rng.choice([15, 30, 45])
    use_rsi = rng.choice([True, False])
    use_bbands = rng.choice([True, False])
    use_vol_confirm = rng.choice([True, False])

    # Build explicit directives
    dir_text = f"""

MANDATORY PER-MODEL DIRECTIVES (for model: {model_id}):
- STRATEGY TYPE = {strat}
- DATA PERIOD = {period_label} using yf.download(..., period="{period_val}", interval="{interval_val}")
- INDICATOR COMBINATION = {combo}
- THRESHOLD STYLE = {thresh}
- LOGIC STYLE = {logic}
- PORTFOLIO STYLE = {port}
- CACHE VARIABLE NAME = {cache_name} (use exactly this name)
- VARIABLE PREFIX = {var_prefix} (prefix all your new variables)
- TARGET ACTION RATE = ~30–50% BUY/SELL under typical conditions

MODEL-SPECIFIC PARAMETERS (use these exact values when applicable):
- FAST_MA = {fast_ma}
- SLOW_MA = {slow_ma}
- RSI_WINDOW = {rsi_win} (use only if `USE_RSI = True`)
- BB_WINDOW = {bb_win} (use only if `USE_BBANDS = True`)
- VOL_LOOKBACK = {vol_lb}
- USE_RSI = {use_rsi}
- USE_BBANDS = {use_bbands}
- USE_VOLUME_CONFIRM = {use_vol_confirm}

Implementation notes:
- Download period/interval exactly as specified
- Use the cache variable name exactly as provided and populate it per ticker
- Prefer these parameter values over any defaults; avoid example values from the prompt
- Ensure all computations check array length and NaN safety before use
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