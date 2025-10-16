import os
import requests
import json
from dotenv import load_dotenv
import time
import hashlib
import random

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


def generate_algorithms_for_agents(selected_agents, ticker, progress_callback=None):
    """Generate algorithms for specific agents via API only (no local fallbacks).
    Returns True only if all agents produced valid code containing execute_trade.
    """
    total = len(selected_agents or [])
    print(f"[gen] Generating algorithms for {total} agents using {ticker} data")
    print(f"[ok] Using {total} models selected from frontend")

    if not selected_agents:
        print("\u274c No models provided for generation")
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
        print(f"\u274c {msg}")
        if progress_callback:
            progress_callback(40, msg)
        return False

    failures = []
    saved = 0
    for i, agent_model in enumerate(selected_agents):
        try:
            step_prog = 30 + int((i / max(1, total)) * 25)  # distribute 30-55%
            if progress_callback:
                progress_callback(step_prog, f"Generating algorithm {i+1}/{total} using {agent_model}...")
            print(f"\nGenerating algorithm {i+1}/{total} using {agent_model}...")

            code = None
            per_model_prompt = base_prompt + build_diversity_directives(agent_model)
            code = generate_algorithm(agent_model, per_model_prompt)
            # Require a proper execute_trade from the model
            if not code or 'def execute_trade' not in code:
                print(f"[error] Missing valid execute_trade in response for {agent_model}")
                failures.append(agent_model)
                continue

            # Emit a short preview snippet to the API progress stream for UX
            if progress_callback and code:
                try:
                    snippet_lines = code.splitlines()[:24]
                    preview = "\n".join(snippet_lines)
                    progress_callback(step_prog, f"PREVIEW::{agent_model}::{preview}")
                except Exception:
                    pass

            _save_code_for_model(code, agent_model)
            saved += 1
        except Exception as e:
            print(f"[error] Error generating algorithm for {agent_model}: {e}")
            failures.append(agent_model)

    # Only fail if NO algorithms were generated successfully
    if saved == 0:
        msg = "Algorithm generation completely failed - no valid algorithms generated"
        print(f"\u274c {msg}")
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

def generate_algorithm(model_id, prompt_text: str):
    """Sends the generation prompt to a specific model and returns its response."""
    print(f"\n--- Generating algorithm with: {model_id} ---")
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": model_id, "messages": [{"role": "user", "content": prompt_text}]}
    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data, timeout=180)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content'].strip()
        
        # Clean up code blocks
        if content.startswith("```python"):
            content = content[9:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        content = content.strip()
        print(f"✅ SUCCESS: Code received from {model_id}.")
        return content
    except requests.exceptions.RequestException as e:
        print(f"❌ FAILED to get code from {model_id}.\n   Error: {e}")
        return None
    except (KeyError, IndexError):
        print(f"❌ FAILED to parse response from {model_id}.")
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