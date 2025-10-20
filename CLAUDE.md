# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Trader Battlefield is a real-time stock market simulation platform where AI-generated trading algorithms compete against each other. The system uses LLMs to generate unique trading strategies, then pits them against each other in a realistic market environment with professional-grade order matching.

**Tech Stack:**
- **Backend**: Python 3.9+, Flask, yfinance, pandas
- **Frontend**: React 18.2.0, React Router 7.8.2
- **LLM Integration**: OpenRouter API (60+ models across 20+ providers)
- **Market Data**: Yahoo Finance (historical OHLCV data)
- **Deployment**: Docker Compose

## Quick Start Commands

### Backend Development

```bash
# Setup virtual environment
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API key
echo "OPENROUTER_API_KEY=your_key_here" > .env

# Run Flask API server
python app.py  # Starts on port 5000

# Run interactive simulation menu
./run_simulation.sh

# Run standalone simulation (without web UI)
python main.py
```

### Frontend Development

```bash
cd frontend
npm install  # or yarn install

# Development server (proxies API to localhost:5000)
npm start  # Starts on port 3000

# Production build
npm run build

# Run tests
npm test
```

### Docker Deployment

```bash
# Build and run both services
docker-compose up --build

# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

## Architecture Overview

### High-Level Data Flow

```
User selects 6 AI models + stock → POST /api/run
    ↓
Background Thread:
  1. Algorithm Generation (open_router/algo_gen.py)
     - Generates 6 unique trading algorithms via OpenRouter API
     - Each model gets deterministic strategy parameters (seeded by name)
     - Saves to generate_algo/generated_algo_*.py
     - Progress callback: "PREVIEW::<model>::<code>"

  2. Market Simulation (main.py → market/market_simulation.py)
     - Loads AlgoAgent wrappers around generated algorithms
     - Creates OrderBook with heap-based price-time priority matching
     - Streams ticks from YFinanceTickGenerator
     - Iterates 100-120 price updates:
       * Get agent decisions (orders)
       * Process through OrderBook matching engine
       * Update portfolios (cash + stock)
       * Expire unfilled limit orders (TTL=1 tick)
     - Calculate ROI and return leaderboard
     - Cleanup: Delete generate_algo/ directory
    ↓
Frontend polls /api/simulation/<sim_id> every 2s
    ↓
Displays: Progress bar, per-model generation status, live code preview, final leaderboard
```

### Critical Components

#### Backend Core (backend/)

**app.py** - Flask API server
- `POST /api/run` - Starts simulation in background thread (returns sim_id)
- `GET /api/simulation/<sim_id>` - Poll for progress/results
- `GET /api/ai_agents` - List available LLM models from ai_agents.json
- `GET /api/data_files` - List available stock CSV files
- Uses threading to prevent blocking on long-running simulations

**main.py** - Simulation orchestrator
- `run_simulation_with_params()` - Coordinates generation + simulation
- `AlgoAgent` class - Wraps AI-generated algorithms with:
  - Deterministic parameters seeded by agent name (md5 hash)
  - Trading frequency throttle (1x, 2x, or 3x per tick)
  - Position sizing (randomized buy/sell fractions)
  - Limit price offsets (basis points)
  - HOLD streak tracking to force action if algorithm too conservative
- Configuration: initial_cash=$10k, max_ticks=100-120, margin=$20k-$40k, short limit=50-100 shares

**market/market_simulation.py** - Core tick-based simulation engine
- `MarketSimulation` class orchestrates:
  - Agent decision gathering
  - Order book matching
  - Portfolio updates (with margin/short selling support)
  - Trade recording and TTL-based order expiration
- **Reservation System**: Prevents over-leveraging by tracking reserved cash/stock per order and per agent
- `SimulationConfig` dataclass for all configurable parameters

**market/order_book.py** - Professional-grade matching engine
- Heap-based price-time priority (bids: max-heap, asks: min-heap)
- Supports LIMIT and MARKET orders
- Partial fills supported
- Returns `Trade` objects with full execution details

**market/tick_generator.py** - Market data provider
- `YFinanceTickGenerator` - Streams historical data from Yahoo Finance
  - Supports multiple timeframes: 1m, 5m, 15m, 30m, 1h, 1d, etc.
  - Supports multiple periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
  - Configurable replay speed via `tick_sleep` parameter
  - Fallback to dummy data if Yahoo Finance unavailable
- `stream()` method yields `TickData` objects (timestamp, OHLC, volume)

**market/agent.py** - Agent framework
- `BaseAgent` - Abstract class requiring `on_tick()` implementation
- Built-in agents: `RandomAgent`, `MomentumAgent`, `MeanReversionAgent`, `MarketMakerAgent`
- `Portfolio` class tracks cash and stock holdings
- `AgentManager` handles agent collection, trade execution, and leaderboard generation

**open_router/algo_gen.py** - LLM algorithm generation
- `generate_algorithms_for_agents()` - Main generation function
- `build_diversity_directives()` - Creates deterministic strategy parameters per model:
  - Strategy type: mean reversion, momentum, volatility, volume-based, technical indicators, etc.
  - Data period: SHORT (5-10d/1m), MEDIUM (15-45d/15m-30m), LONG (60-90d/30m-1h)
  - Indicator combination: single, dual crossover, multi-factor, custom
  - Threshold style: aggressive, moderate, conservative, adaptive
  - Specific parameters: FAST_MA, SLOW_MA, RSI_WINDOW, BB_WINDOW, VOL_LOOKBACK, etc.
- Validates generated code has `execute_trade(ticker, cash_balance, shares_held) -> str` function
- Returns: "BUY", "SELL", or "HOLD"
- Enforces error handling (NaN checks, division-by-zero, array length validation)

#### Frontend Core (frontend/src/)

**App.js** - React Router setup
- Routes: / (Dashboard), /models (Model showcase), /about, /contact

**components/Dashboard.js** - Main simulation interface
- State management for 6 agent slots, stock selection, progress, results
- `handleStartSimulation()` - Validates 6 agents selected, POSTs to /api/run
- `pollSimulationStatus()` - Polls every 2s for progress updates
- Parses "PREVIEW::<model>::<code>" for live code preview during generation
- Volume multiplier (0.5x-5x) and aggressive mode toggles
- Algorithm preview modal with copy/download functionality

**components/CustomDropdown.js** - Agent selector with provider grouping
- Prevents duplicate selections across 6 slots
- Shows provider icons

**components/ModelDirectory.js** - Model list grouped by provider
- Compact mode for dashboard, full mode for /models page

### Key Design Patterns

#### 1. Deterministic Diversity in Algorithm Generation
Each AI model receives unique strategy directives seeded by model name hash. This ensures:
- Different models don't generate identical strategies
- Reproducible results (same model → same strategy type)
- Coverage of diverse trading approaches

#### 2. Reservation Ledger for Order Management
The simulation tracks reserved cash/stock at two levels:
- **Per-order**: `_order_reserved_cash`, `_order_reserved_stock`
- **Per-agent**: `_agent_reserved_cash`, `_agent_reserved_stock`

This prevents agents from over-leveraging without blocking orders from being placed.

#### 3. Progress Streaming via Callbacks
`progress_callback()` supports special message formats:
- `"PREVIEW::<model>::<code>"` - Live algorithm code preview
- `"Generating algorithm for <model>..."` - Generation status
- Regular status messages for simulation progress

#### 4. Agent Wrapper Pattern (AlgoAgent)
The `AlgoAgent` class wraps dynamically imported trading algorithms and adds:
- Trading frequency throttling to prevent spam
- Position sizing randomization for diversity
- HOLD streak detection to force action
- Exception handling around algorithm execution

## Data Storage

### Stock Data (backend/data/)
CSV files with OHLCV format:
- AAPL_data.csv, MSFT_data.csv, GOOGL_data.csv, AMZN_data.csv
- NVDA_data.csv, TSLA_data.csv, META_data.csv, NFLX_data.csv
- stock_data.csv (large historical dataset)

Structure:
```csv
Date,Open,High,Low,Close,Volume
YYYY-MM-DD,price,price,price,price,volume
```

### AI Models Configuration (backend/open_router/ai_agents.json)
JSON mapping provider → list of model IDs
- 60+ models across 20+ providers
- Loaded by frontend via `/api/ai_agents`
- Used for algorithm generation

### Generated Algorithms (backend/generate_algo/)
Temporary directory created during simulation
- Contains `generated_algo_<model_name>.py` files
- Each file has `execute_trade(ticker, cash_balance, shares_held)` function
- Deleted after simulation completes

## Configuration

### Environment Variables (backend/.env)
```bash
OPENROUTER_API_KEY=sk-or-v1-...
PORT=5000
FLASK_ENV=development
PYTHONUNBUFFERED=1
```

### Simulation Parameters (main.py)
```python
max_ticks = 120 if aggressive_mode else 100
initial_cash = 10000.0
initial_stock = 5-8  # Seeded per agent
cash_borrow_limit = 40000.0 if aggressive_mode else 20000.0
max_short_shares = 100 if aggressive_mode else 50
order_ttl_ticks = 1  # Limit orders expire each tick
tick_sleep = 0.01  # Simulation speed
volume_multiplier = 0.5-5.0  # Scales order quantities
```

## Common Development Tasks

### Adding a New Built-in Trading Agent

1. Edit `backend/market/agent.py`
2. Create subclass of `BaseAgent`
3. Implement `on_tick(self, price, current_tick, cash=0.0, stock=0)` method
4. Return list of order dicts: `[{"agent": self.name, "side": "buy/sell", "price": float, "quantity": int}]`

Example:
```python
class MyAgent(BaseAgent):
    def on_tick(self, price, current_tick, cash=0.0, stock=0):
        if current_tick % 10 == 0 and cash > price:
            return [{"agent": self.name, "side": "buy",
                    "price": price, "quantity": 1}]
        return []
```

### Modifying Algorithm Generation Prompts

Edit `backend/open_router/algo_gen.py`:
- `build_generation_prompt()` - Base prompt template
- `build_diversity_directives()` - Per-model strategy parameters

### Adding New Stock Data

1. Place CSV file in `backend/data/`
2. Ensure format matches: `Date,Open,High,Low,Close,Volume`
3. Frontend will auto-discover via `/api/data_files`

### Extending Frontend UI

- Dashboard controls: Edit `frontend/src/components/Dashboard.js`
- New routes: Edit `frontend/src/App.js`
- Styling: CSS files in `frontend/src/components/`

## Testing

### Backend Tests
```bash
cd backend
source venv/bin/activate

# Test individual components via run_simulation.sh
./run_simulation.sh
# Select option 5: Test system components

# Or test specific modules
python -c "from market.order_book import OrderBook; print('OrderBook OK')"
python -c "from market.tick_generator import YFinanceTickGenerator; print('TickGen OK')"
```

### Frontend Tests
```bash
cd frontend
npm test  # Runs Jest test suite
```

### End-to-End Test
```bash
# Terminal 1: Start backend
cd backend && source venv/bin/activate && python app.py

# Terminal 2: Start frontend
cd frontend && npm start

# Browser: http://localhost:3000
# Select 6 agents, choose stock, click START
```

## Important Notes

### Algorithm Generation
- Each generated algorithm must have `execute_trade(ticker, cash_balance, shares_held)` function
- Must return exactly: "BUY", "SELL", or "HOLD"
- Must use `yfinance` with `progress=False` to avoid cluttering output
- Must handle NaN values and check array lengths before indexing
- Should use module-level cache dict to avoid re-fetching data every tick

### Order Book Behavior
- Orders are matched using price-time priority
- Market orders execute immediately at best available price
- Limit orders rest on the book until matched or expired
- Partial fills are supported (order can be executed multiple times)

### Margin and Short Selling
- Agents can borrow cash up to `cash_borrow_limit` (default $20k-$40k)
- Agents can short up to `max_short_shares` (default 50-100 shares)
- Reservation system prevents over-leveraging
- Negative cash is allowed up to borrow limit

### Cleanup
- Generated algorithm files are deleted after each simulation
- Use try/finally blocks to ensure cleanup happens even on errors

## Troubleshooting

**ModuleNotFoundError: No module named 'X'**
→ Activate venv and install requirements: `source venv/bin/activate && pip install -r requirements.txt`

**No generated algorithms found**
→ Check OpenRouter API key in `.env` file
→ Run `python -c "from open_router.algo_gen import generate_algorithms_for_agents; print('OK')"`

**Yahoo Finance connection errors**
→ System falls back to dummy data automatically
→ Check internet connection
→ Try shorter periods: `period="1d"` instead of `period="30d"`

**Frontend proxy not working**
→ Ensure backend is running on port 5000
→ Check `"proxy": "http://localhost:5000"` in `frontend/package.json`

**Simulation hangs**
→ Check for infinite loops in generated algorithms
→ Reduce `max_ticks` in simulation config
→ Add timeouts around algorithm execution
