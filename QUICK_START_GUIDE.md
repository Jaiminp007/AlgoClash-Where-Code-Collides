# AI Trader Battlefield - Quick Start Guide

## 🎯 What We're Building

Transform your current system into a **live trading simulation platform** where AI agents compete using real market data, adapt their strategies every 3 hours, and battle for the best ROI.

---

## 📊 Current vs New System

| Feature | Current | New |
|---------|---------|-----|
| **Data Source** | Static CSV files | Live Finnhub API |
| **Timing** | Real-time | Day-delayed (uses yesterday's data) |
| **Order Execution** | Complex order book | Instant open market |
| **AI Learning** | Static algorithms | Adaptive (updates every 3 hours) |
| **Checkpoints** | End only | Every 3 hours (8 total) |
| **Metrics** | ROI only | ROI + Sharpe + Max Drawdown + Risk |
| **Database** | None | MongoDB |

---

## 🏗️ Architecture Overview

```
Today (Day N)
    ↓
Python Script: Fetch Day N-1 data from Finnhub
    ↓
Store in MongoDB Database
    ↓
AI Agents receive data
    ↓
┌─────────────────────────────────────┐
│   24-Hour Simulation (8 Checkpoints) │
├─────────────────────────────────────┤
│  Every 3 hours:                     │
│  1. Compare agent vs actual market  │
│  2. Calculate performance metrics   │
│  3. Regenerate improved algorithm   │
│  4. Continue trading                │
└─────────────────────────────────────┘
    ↓
Final Results:
  - ROI Ranking
  - Sharpe Ratio
  - Max Drawdown
  - Risk Measures
    ↓
Winner Declared! 🏆
```

---

## ⚡ 10-Week Implementation Plan

### **Week 1: MongoDB Setup**
```bash
# Install MongoDB
docker run -d --name ai_trader_mongodb -p 27017:27017 mongo:7.0

# Create Python repositories
backend/database/
  ├── mongo_config.py          # Connection manager
  ├── market_data_repository.py
  ├── simulation_repository.py
  └── checkpoint_repository.py

# Test connection
python backend/test_mongo_connection.py
```

**Goal:** MongoDB running + repositories working

---

### **Week 2: Finnhub Integration**
```bash
# Sign up: https://finnhub.io
# Get API key (free tier: 60 calls/min)

# Create standalone script
backend/scripts/finnhub_to_mongo.py

# Run it
python backend/scripts/finnhub_to_mongo.py --ticker AAPL

# Verify in MongoDB
docker exec -it ai_trader_mongodb mongosh
> use ai_trader_battlefield
> db.market_data.countDocuments({ ticker: "AAPL" })
```

**Goal:** Can fetch Day N-1 data and store in MongoDB

---

### **Week 3: Simplify Order Execution**
```python
# Remove order book complexity
# backend/market/market_simulation.py

def _process_tick(self, tick, current_tick):
    current_price = tick.close

    for agent_name, agent in agents.items():
        decision = agent.on_tick(price, tick, cash, shares)

        if decision == "BUY":
            # Instant execution at market price
            cash -= quantity * current_price
            shares += quantity

        elif decision == "SELL":
            # Instant execution at market price
            cash += quantity * current_price
            shares -= quantity
```

**Goal:** Simplified instant market execution

---

### **Week 4: Checkpoint System**
```python
# backend/market/checkpoint_manager.py

class CheckpointManager:
    def __init__(self, total_ticks=288):  # 24 hours at 5-min intervals
        self.checkpoint_interval = total_ticks // 8  # Every 3 hours

    def is_checkpoint(self, current_tick):
        return current_tick % self.checkpoint_interval == 0

    def evaluate_checkpoint(self, agent, portfolio, tick_history):
        # Compare agent performance vs actual market
        # Calculate ROI, prediction accuracy, performance delta
        # Store in MongoDB
        return evaluation
```

**Goal:** Detect checkpoints every 3 hours

---

### **Week 5: Adaptive Algorithm Generation**
```python
# backend/open_router/adaptive_algo_gen.py

def generate_updated_algorithm(checkpoint_eval):
    prompt = f"""
    Your Performance:
    - ROI: {checkpoint_eval.roi}%
    - Prediction Accuracy: {checkpoint_eval.prediction_accuracy}%
    - vs Market: {checkpoint_eval.performance_delta}%

    Market Reality (Last 3 Hours):
    - High: ${checkpoint_eval.actual_high}
    - Low: ${checkpoint_eval.actual_low}

    Generate IMPROVED algorithm that:
    1. Fixes underperformance issues
    2. Uses better technical indicators
    3. Optimizes position sizing
    """

    # Call OpenRouter API
    new_code = openrouter_api.generate(prompt)
    return new_code
```

**Goal:** AI algorithms that learn and improve

---

### **Week 6: Advanced Metrics**
```python
# backend/market/performance_metrics.py

class PerformanceMetrics:
    @staticmethod
    def calculate_sharpe_ratio(returns):
        return mean(returns) / std(returns)

    @staticmethod
    def calculate_max_drawdown(portfolio_values):
        peak = portfolio_values[0]
        max_dd = 0
        for value in portfolio_values:
            if value > peak: peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        return max_dd * 100
```

**Goal:** Professional trading metrics implemented

---

### **Week 7: Daily Automation**
```python
# backend/daily_runner.py

def run_daily_simulation():
    # 1. Fetch Day N-1 from Finnhub → MongoDB
    fetcher.fetch_and_store("AAPL")

    # 2. Load 6 AI models
    models = ["gpt-4", "gemini-pro", "claude-3", ...]

    # 3. Run 24-hour simulation with checkpoints
    results = simulation.run(models)

    # 4. Store results in MongoDB
    db.store_results(results)

    print(f"Winner: {results['winner']}")

# Run daily at 9 AM
schedule.every().day.at("09:00").do(run_daily_simulation)
```

**Goal:** Fully automated daily runs

---

### **Week 8: Frontend Updates**
```jsx
// frontend/src/components/CheckpointTimeline.js

function CheckpointTimeline({ checkpoints }) {
    return (
        <div className="checkpoint-timeline">
            {checkpoints.map((cp, i) => (
                <div key={i} className="checkpoint">
                    <h3>Checkpoint {cp.number} - {cp.hours}h</h3>
                    <p>ROI: {cp.roi}%</p>
                    <p>Accuracy: {cp.prediction_accuracy}%</p>
                    <button onClick={() => viewAlgorithm(cp)}>
                        View Updated Algorithm
                    </button>
                </div>
            ))}
        </div>
    )
}
```

**Goal:** Visualize checkpoints and algorithm evolution

---

### **Week 9: Testing**
```bash
# Unit tests
pytest backend/tests/test_checkpoint_manager.py
pytest backend/tests/test_performance_metrics.py
pytest backend/tests/test_finnhub_provider.py

# Integration test
python backend/daily_runner.py --test-mode

# End-to-end test
docker-compose up
# Browser: http://localhost:3000
# Run simulation and verify all 8 checkpoints work
```

**Goal:** Everything works reliably

---

### **Week 10: Deploy**
```bash
# Update docker-compose.yml
docker-compose up --build

# Set up MongoDB Atlas (cloud)
# https://www.mongodb.com/cloud/atlas

# Set up cron job
0 9 * * * cd /path/to/project && python backend/daily_runner.py

# Monitor logs
tail -f logs/simulation.log
```

**Goal:** Production-ready system

---

## 🚀 Quick Commands Reference

### **MongoDB**
```bash
# Start MongoDB
docker-compose up -d mongodb

# View data
docker exec -it ai_trader_mongodb mongosh ai_trader_battlefield
> db.market_data.find({ ticker: "AAPL" }).limit(5)
> db.checkpoints.countDocuments()
```

### **Fetch Data**
```bash
# Fetch previous trading day
python backend/scripts/finnhub_to_mongo.py --ticker AAPL

# Fetch specific date
python backend/scripts/finnhub_to_mongo.py --ticker AAPL --date 2025-10-15

# Fetch last 5 days
python backend/scripts/finnhub_to_mongo.py --ticker AAPL --days-back 5

# Multiple tickers
for ticker in AAPL MSFT GOOGL AMZN; do
    python backend/scripts/finnhub_to_mongo.py --ticker $ticker
done
```

### **Run Simulation**
```bash
# Development
cd backend && python daily_runner.py

# Production (Docker)
docker-compose up
```

### **Test Everything**
```bash
# Backend tests
cd backend && pytest

# Check MongoDB connection
python backend/test_mongo_connection.py

# Verify Finnhub API
python backend/scripts/finnhub_to_mongo.py --ticker AAPL --days-back 1
```

---

## 📦 Dependencies to Install

**Backend (requirements.txt):**
```
# Existing
pandas
numpy
yfinance
matplotlib
requests
gunicorn
python-dotenv
Flask
Flask-Cors

# NEW
pymongo==4.6.1          # MongoDB driver
motor==3.3.2            # Async MongoDB (optional)
dnspython==2.4.2        # MongoDB Atlas support
finnhub-python==2.4.19  # Finnhub API
schedule==1.2.0         # Daily scheduling
```

**Install:**
```bash
cd backend
pip install -r requirements.txt
```

---

## 🔑 Environment Variables (.env)

```bash
# Finnhub API (free tier)
FINNHUB_API_KEY=your_key_from_finnhub_io

# OpenRouter API (for AI algorithm generation)
OPENROUTER_API_KEY=sk-or-v1-your_key_here

# MongoDB (local)
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=ai_trader_battlefield

# MongoDB Atlas (cloud - optional)
# MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/

# Flask
PORT=5000
FLASK_ENV=development
PYTHONUNBUFFERED=1
```

---

## 📊 MongoDB Collections

**5 Collections:**

1. **market_data** - OHLCV ticks from Finnhub
   ```javascript
   { ticker: "AAPL", timestamp: ISODate(), open, high, low, close, volume, vwap }
   ```

2. **simulation_sessions** - Trading session metadata
   ```javascript
   { session_id, ticker, start_time, status, config, agents: [...] }
   ```

3. **agent_performance** - Final results
   ```javascript
   { session_id, agent_name, roi, sharpe_ratio, max_drawdown, rank }
   ```

4. **checkpoints** - 3-hour evaluations
   ```javascript
   {
     session_id, agent_name, checkpoint_number,
     agent_state: { portfolio_value, cash, shares, roi },
     market_data: { high, low, close, volume, vwap },
     algorithm: { version, code, update_reason }
   }
   ```

5. **trades** - Individual executions
   ```javascript
   { session_id, agent_name, timestamp, side: "BUY/SELL", quantity, price }
   ```

---

## 🎯 Success Criteria

✅ **Technical:**
- MongoDB storing all data
- Finnhub script fetches Day N-1 successfully
- Simulation runs 24 hours with 8 checkpoints
- Algorithms regenerate at each checkpoint
- Advanced metrics calculated correctly

✅ **Business:**
- Clear winner based on composite score
- Adaptive algorithms outperform static ones
- Sharpe ratio improves over checkpoints
- Max drawdown stays within acceptable limits
- System runs daily without manual intervention

---

## 💡 Key Innovations

### **1. Adaptive Learning**
AI agents don't just trade—they **learn and evolve** every 3 hours based on performance feedback.

### **2. Real Market Validation**
Uses actual Day N-1 data, so agents are tested against **real market conditions**.

### **3. No Order Book Complexity**
Simplified to instant execution = faster development, easier debugging.

### **4. MongoDB Flexibility**
NoSQL schema allows storing algorithm code, nested checkpoint data, and evolving metrics without migrations.

### **5. Standalone Data Fetcher**
`finnhub_to_mongo.py` can run independently via cron, separate from main simulation.

---

## 🐛 Common Issues & Solutions

**Issue:** Can't connect to MongoDB
**Solution:**
```bash
docker ps  # Check if MongoDB container is running
docker-compose up -d mongodb
```

**Issue:** Finnhub API rate limit
**Solution:** Free tier = 60 calls/min. Add delays:
```python
import time
for ticker in tickers:
    fetch_and_store(ticker)
    time.sleep(2)  # 2-second delay
```

**Issue:** Algorithm generation fails
**Solution:** Check OpenRouter API key and credits:
```bash
curl https://openrouter.ai/api/v1/auth/key \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

**Issue:** Checkpoint not triggering
**Solution:** Verify tick count:
```python
total_ticks = 288  # 24 hours * 12 (5-min intervals)
checkpoint_interval = total_ticks // 8  # Should be 36
```

---

## 📈 Expected Results

After full implementation, you'll see:

```
=== SIMULATION RESULTS ===
Session: 2025-10-16 | Ticker: AAPL

🏆 WINNER: GPT-4 Turbo
  - ROI: 18.42%
  - Sharpe Ratio: 2.34
  - Max Drawdown: 4.21%
  - Composite Score: 87.6

📊 TOP 3:
  1. GPT-4 Turbo      ROI: 18.42%  Sharpe: 2.34  Drawdown: 4.21%
  2. Claude 3 Sonnet  ROI: 15.67%  Sharpe: 2.01  Drawdown: 5.89%
  3. Gemini Pro       ROI: 12.34%  Sharpe: 1.78  Drawdown: 6.45%

🔄 CHECKPOINT EVOLUTION:
  CP1 (3h):  ROI 2.1%  → Algorithm updated: "Too conservative"
  CP2 (6h):  ROI 5.4%  → Algorithm updated: "Improve timing"
  CP3 (9h):  ROI 8.9%  → Algorithm updated: "Increase aggression"
  CP4 (12h): ROI 11.2% → Algorithm updated: "Optimize entries"
  CP5 (15h): ROI 13.8% → Algorithm updated: "Fine-tune exits"
  CP6 (18h): ROI 15.6% → Algorithm updated: "Maintain edge"
  CP7 (21h): ROI 17.1% → Algorithm updated: "Lock profits"
  CP8 (24h): ROI 18.4% → FINAL
```

---

## 🔗 Important Links

- **Finnhub API:** https://finnhub.io/docs/api
- **MongoDB Atlas:** https://www.mongodb.com/cloud/atlas
- **OpenRouter API:** https://openrouter.ai/docs
- **Full Plan:** [OPTIMIZATION_PLAN.md](./OPTIMIZATION_PLAN.md)
- **CLAUDE.md:** [CLAUDE.md](./CLAUDE.md)

---

## 🚦 Getting Started RIGHT NOW

**Option 1: Fastest (Local)**
```bash
# 1. Start MongoDB
docker run -d --name ai_trader_mongodb -p 27017:27017 mongo:7.0

# 2. Install dependencies
cd backend
pip install pymongo finnhub-python

# 3. Set API key
export FINNHUB_API_KEY=your_key_here

# 4. Test fetch
python scripts/finnhub_to_mongo.py --ticker AAPL

# ✅ You now have live market data in MongoDB!
```

**Option 2: Docker Everything**
```bash
# 1. Update docker-compose.yml (see OPTIMIZATION_PLAN.md)
# 2. Start all services
docker-compose up --build

# ✅ Full stack running!
```

---

## 📞 Need Help?

1. Check [OPTIMIZATION_PLAN.md](./OPTIMIZATION_PLAN.md) for detailed implementation
2. Check [CLAUDE.md](./CLAUDE.md) for codebase guide
3. Review logs: `tail -f logs/simulation.log`
4. Test MongoDB: `python backend/test_mongo_connection.py`

---

**Ready to build the future of AI trading? Let's go! 🚀**
