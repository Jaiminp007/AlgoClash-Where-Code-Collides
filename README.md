# AI Trader Battlefield

An experimental platform where AI-generated trading algorithms compete against each other in a simulated stock market. Multiple Large Language Models (LLMs) automatically generate trading strategies, which then battle in a 5-minute trading session to see which algorithm achieves the highest return on investment (ROI).

---

## 📐 Architecture Overview
<img width="1156" height="501" alt="Screenshot 2025-07-25 at 11 57 23 AM" src="https://github.com/user-attachments/assets/b4f4d481-9d2d-4b2c-9e56-b36efd55736a" />

---

> All agents start with the same capital and compete in real time as prices shift based on their combined trading behavior.

---

## 🧠 Description

This platform demonstrates the intersection of AI code generation and algorithmic trading by:

1. **Using LLMs to generate trading algorithms** - Select from 50+ AI models (Claude, Gemini, GPT, Llama, etc.) to automatically generate Python trading functions
2. **Simulating realistic market conditions** - A custom order book and tick-based engine processes trades with price-time priority matching
3. **Competing strategies head-to-head** - 2-6 AI-generated algorithms compete simultaneously on the same stock data
4. **Visualizing performance in real-time** - Interactive dashboard shows live leaderboards, portfolio values, and market charts

Each generated algorithm receives tick-by-tick market data and makes autonomous BUY/SELL/HOLD decisions. The simulation engine processes these orders through a realistic order-matching system, and the agent with the highest ROI at session end wins.

---

## 🌟 Core Features

### 🤖 AI-Powered Algorithm Generation
- **50+ LLM Integration** - Generate strategies using Claude Opus, Gemini, GPT-4, Llama, DeepSeek, Mistral, and more via OpenRouter API
- **Automatic Code Creation** - LLMs write complete Python trading functions based on strategy prompts
- **Algorithm Preview** - Review and inspect generated code before running simulations
- **Multi-Model Comparison** - Test which AI model creates the most profitable strategies

### 🏛 Advanced Market Simulation
- **Order Book Engine** - Professional-grade order matching with price-time priority
- **Realistic Tick Data** - Historical data for AAPL, GOOGL, TSLA, MSFT, AMZN, NVDA, META, NFLX
- **60-Tick Sessions** - Each battle runs for 60 market ticks (~5 minutes real-time)
- **Multiple Agent Support** - Run 2-6 competing algorithms simultaneously

### 📈 Interactive Dashboard
- **Live Leaderboard** - Real-time ROI rankings updated every tick
- **Market Charts** - Dynamic price visualization with Recharts
- **Performance Metrics** - Track P&L, win rate, max drawdown, and portfolio value
- **Trade History** - Complete audit log of all executed orders

---

## 🏆 Scoring System

Each trading session lasts **60 ticks** (approximately 5 minutes). Agents are ranked by:

| Metric              | Description                                  |
|---------------------|----------------------------------------------|
| **ROI (%)**         | Return on investment - primary ranking metric |
| **P&L**             | Absolute profit or loss in dollars           |
| **Win Rate (%)**    | Percentage of profitable trades              |
| **Max Drawdown (%)** | Largest drop from peak portfolio value      |
| **Portfolio Value** | Total cash + stock holdings at market price  |
| **Trade Count**     | Total number of executed trades              |

> The agent with the **highest ROI** wins the session.

---

## 🛠️ Tech Stack

### Backend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | Flask | REST API server |
| **Language** | Python 3.11+ | Core simulation logic |
| **LLM Integration** | OpenRouter API | Access to 50+ AI models |
| **Market Data** | Yahoo Finance (yfinance) | Historical stock data |
| **Data Processing** | Pandas, NumPy | Data analysis |
| **Server** | Gunicorn | Production WSGI server |

### Frontend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | React 18.2 | User interface |
| **Routing** | React Router v7 | Page navigation |
| **Styling** | Tailwind CSS | Responsive design |
| **Charts** | Recharts 3.3 | Market visualization |
| **Animation** | Framer Motion 12 | Smooth transitions |

### Deployment
- **Frontend Hosting** - Vercel
- **Backend Server** - Flask with Gunicorn
- **No Database** - In-memory simulation state

---

## 🚀 Live Demo

The platform is live at: **[ai-trader-battlefield-fro.onrender.com](https://ai-trader-battlefield-fro.onrender.com)**

---

## ⚙️ How It Works

### 1. Algorithm Generation
```
User selects AI models → OpenRouter API generates trading functions → Code saved as Python files
```

Each generated algorithm implements:
```python
def execute_trade(ticker: str, cash_balance: float, shares_held: int) -> str:
    # AI-generated trading logic
    return "BUY" | "SELL" | "HOLD"
```

### 2. Simulation Engine
```
Load tick data → For each tick:
  - Call each agent's execute_trade()
  - Collect orders (BUY/SELL/HOLD)
  - Match orders in order book
  - Update prices based on trades
  - Calculate portfolio values
→ Return final rankings
```

### 3. Order Matching
- **Price-time priority** - Best price gets filled first; ties broken by timestamp
- **Partial fills supported** - Large orders can be partially executed
- **Bid-ask spread** - Realistic market microstructure
- **No short selling** - Long-only strategies

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ai_agents` | GET | List available AI models |
| `/api/data_files` | GET | List available stock datasets |
| `/api/algos` | GET | List generated algorithms |
| `/api/algos/{filename}` | GET | Preview algorithm code |
| `/api/generate` | POST | Generate new algorithms |
| `/api/run` | POST | Start simulation |
| `/api/status/{sim_id}` | GET | Get simulation status |

---

## 📁 Project Structure

```
algoclash-v1-test/
├── backend/
│   ├── app.py                    # Flask server & REST API
│   ├── market/                   # Simulation engine
│   │   ├── market_simulation.py  # Session orchestrator
│   │   ├── order_book.py         # Order matching engine
│   │   ├── tick_generator.py     # Stock data provider
│   │   └── agent.py              # Trading agent manager
│   ├── open_router/              # LLM integration
│   │   ├── algo_gen.py           # Algorithm generator
│   │   ├── model_fecthing.py     # Fetch AI models
│   │   └── ai_agents.json        # 50+ model configs
│   ├── data/                     # Historical stock CSVs
│   └── generate_algo/            # Generated Python algorithms
├── frontend/
│   ├── src/
│   │   ├── components/           # 16 React components
│   │   │   ├── Dashboard.js      # Main battle arena
│   │   │   ├── ResultsDashboard.js  # Leaderboard
│   │   │   ├── Models.js         # AI model directory
│   │   │   └── ...
│   │   └── App.js
│   └── package.json
└── README.md
```

---

## 🎯 Use Cases

### What This IS
- Educational platform for algorithmic trading concepts
- AI code generation benchmark (which LLM writes better trading code?)
- Strategy testing environment with realistic market simulation
- Interactive demonstration of market dynamics

### What This IS NOT
- Production trading system (no real money or live markets)
- Financial advice or investment tool
- High-frequency trading platform
- Connection to real brokerages

---

## ⚡ Configuration

### Simulation Parameters
```python
SimulationConfig(
    max_ticks=60,              # Session length (60 ticks)
    tick_sleep=1.0,            # Seconds between ticks
    initial_cash=10000.0,      # Starting capital per agent
    enable_order_book=True,    # Realistic order matching
    allow_short=False,         # No short selling
    allow_negative_cash=False  # No margin trading
)
```

### Supported Stocks
- AAPL (Apple)
- GOOGL (Google)
- TSLA (Tesla)
- MSFT (Microsoft)
- AMZN (Amazon)
- NVDA (NVIDIA)
- META (Meta)
- NFLX (Netflix)

---

## 🤝 Contributing

Contributions are welcome! This is an experimental platform under active development.

**Current Branch:** `v1`
**Main Branch:** `main`

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenRouter** - For providing unified access to 50+ LLM providers
- **Yahoo Finance** - For historical market data
- All open-source libraries that made this project possible

---

**Built with Flask, React, and AI curiosity.**


