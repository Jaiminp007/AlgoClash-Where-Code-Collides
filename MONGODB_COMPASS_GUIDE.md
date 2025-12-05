# MongoDB Compass Setup Guide for AlgoClash

This guide will help you view your AlgoClash data in MongoDB Compass (GUI).

## 📍 Database Configuration

**Connection String:** `mongodb://localhost:27017/ai_trader_battlefield`

**Database Name:** `ai_trader_battlefield`

All data is stored in this database and can be viewed in MongoDB Compass.

---

## 🚀 Step 1: Install MongoDB Compass

### Download MongoDB Compass

**Official Download:** https://www.mongodb.com/try/download/compass

Or install via command line:

**Mac:**
```bash
brew install --cask mongodb-compass
```

**Windows:**
Download from: https://www.mongodb.com/try/download/compass

**Linux:**
```bash
wget https://downloads.mongodb.com/compass/mongodb-compass_latest_amd64.deb
sudo dpkg -i mongodb-compass_latest_amd64.deb
```

---

## 🔌 Step 2: Connect to Your Database

### Method 1: Quick Connect (Recommended)

1. **Open MongoDB Compass**

2. **In the connection screen, paste:**
   ```
   mongodb://localhost:27017/ai_trader_battlefield
   ```

3. **Click "Connect"**

4. **You should see:**
   - Database: `ai_trader_battlefield`
   - Collections listed in the sidebar

### Method 2: Manual Configuration

1. **Open MongoDB Compass**

2. **Click "New Connection"**

3. **Enter details:**
   - **Host:** `localhost`
   - **Port:** `27017`
   - **Authentication:** None (local development)
   - **Default Database:** `ai_trader_battlefield`

4. **Click "Connect"**

---

## 📊 Step 3: View Your Data

Once connected, you'll see the `ai_trader_battlefield` database with the following collections:

### 📁 Collections Structure

#### **Stock Data Collections (2 per stock):**

For each stock ticker (AAPL, AMZN, GME, etc.):

```
📁 AAPL_historical
   └── Daily stock data (for algorithm generation)
   └── Fields: date, open, high, low, close, volume, ticker, fetched_at

📁 AAPL_simulation
   └── Minute-by-minute data (for market simulation)
   └── Fields: datetime, open, high, low, close, volume, ticker, simulation_date, fetched_at
```

**Repeat for each ticker:** AMZN, GME, GOOGL, META, MSFT, NFLX, NVDA, TSLA, BAC, ORCL

#### **Application Data Collections:**

```
📁 generations
   └── Algorithm generation requests
   └── Fields: generation_id, selected_models, selected_stock, status, algorithms, etc.

📁 simulations
   └── Market simulation runs
   └── Fields: simulation_id, generation_id, stock_ticker, results, leaderboard, winner, etc.

📁 algorithms
   └── Individual algorithm code and performance
   └── Fields: simulation_id, model_name, code, performance_roi, etc.

📁 simulation_ticks
   └── Tick-by-tick simulation data
   └── Fields: simulation_id, tick_number, price, agent_portfolios, trades, etc.
```

---

## 🔍 Step 4: Explore Your Data

### **View Historical Stock Data (AAPL example):**

1. Click on **`AAPL_historical`** collection
2. You'll see daily stock data
3. Click on any document to expand and see:
   - Date
   - Open, High, Low, Close prices
   - Volume
   - Ticker symbol

**Example Document:**
```json
{
  "_id": ObjectId("..."),
  "date": "2024-11-21",
  "open": 175.23,
  "high": 178.45,
  "low": 174.12,
  "close": 177.89,
  "volume": 52341234,
  "ticker": "AAPL",
  "data_type": "historical",
  "fetched_at": ISODate("2024-11-25T...")
}
```

### **View Simulation Data (Minute-by-Minute):**

1. Click on **`AAPL_simulation`** collection
2. You'll see minute-level data for the trading day
3. Each document represents one minute of trading

**Example Document:**
```json
{
  "_id": ObjectId("..."),
  "datetime": "2024-11-22 09:30:00",
  "open": 177.50,
  "high": 177.75,
  "low": 177.45,
  "close": 177.60,
  "volume": 125000,
  "ticker": "AAPL",
  "data_type": "simulation",
  "simulation_date": "2024-11-22",
  "fetched_at": ISODate("2024-11-25T...")
}
```

### **View Algorithm Generations:**

1. Click on **`generations`** collection
2. See all algorithm generation requests
3. Each document contains:
   - Which models were used
   - Generated algorithm code
   - Generation status

**Example Document:**
```json
{
  "_id": ObjectId("..."),
  "generation_id": "abc-123-def",
  "selected_models": ["gpt-4", "claude-haiku"],
  "selected_stock": "AAPL_data.csv",
  "status": "completed",
  "progress": 100,
  "algorithms": {
    "gpt-4": "def execute_trade(...):\n    ...",
    "claude-haiku": "def execute_trade(...):\n    ..."
  },
  "created_at": ISODate("2024-11-25T..."),
  "updated_at": ISODate("2024-11-25T...")
}
```

### **View Simulation Results:**

1. Click on **`simulations`** collection
2. See completed market simulations
3. View leaderboards and winners

**Example Document:**
```json
{
  "_id": ObjectId("..."),
  "simulation_id": "sim-456-xyz",
  "generation_id": "abc-123-def",
  "stock_ticker": "AAPL",
  "status": "completed",
  "leaderboard": [
    {
      "name": "gpt-4",
      "roi": 0.15,
      "current_value": 11500,
      "trades": 42
    },
    {
      "name": "claude-haiku",
      "roi": 0.08,
      "current_value": 10800,
      "trades": 35
    }
  ],
  "winner": {
    "name": "gpt-4",
    "roi": 0.15,
    "current_value": 11500
  },
  "created_at": ISODate("2024-11-25T..."),
  "completed_at": ISODate("2024-11-25T...")
}
```

---

## 🛠️ Useful MongoDB Compass Features

### **1. Filter Documents**

In the filter bar, you can query specific data:

**Examples:**

```json
// Get all AAPL historical data for November
{"ticker": "AAPL", "date": {$regex: "^2024-11"}}

// Get simulations that are completed
{"status": "completed"}

// Get high-performing algorithms (ROI > 10%)
{"leaderboard.roi": {$gt: 0.10}}
```

### **2. Sort Documents**

Click on column headers to sort:
- Sort by `date` (ascending/descending)
- Sort by `created_at` to see newest first
- Sort by `roi` to see best performers

### **3. Aggregation Pipeline**

Use the Aggregation tab for complex queries:

**Example: Get average closing price for AAPL**

```json
[
  {$match: {ticker: "AAPL"}},
  {$group: {
    _id: null,
    avgClose: {$avg: "$close"},
    maxClose: {$max: "$close"},
    minClose: {$min: "$close"}
  }}
]
```

### **4. Export Data**

1. Select a collection
2. Click "Export Collection"
3. Choose format (JSON or CSV)
4. Save to your computer

### **5. Schema Analysis**

Click on "Schema" tab to see:
- Field types
- Data distribution
- Missing fields
- Field uniqueness

---

## 📋 Quick Reference

### **After Fetching Data:**

```bash
# Fetch minute-level stock data
python fetch_minute_data_mongodb.py
```

**Collections Created:**
- ✅ AAPL_historical (daily data)
- ✅ AAPL_simulation (minute data)
- ✅ AMZN_historical
- ✅ AMZN_simulation
- ✅ ... (for all 11 stocks)

### **After Running Simulations:**

```bash
# Run Flask app
python app.py
```

**Collections Updated:**
- ✅ generations (algorithm generation requests)
- ✅ simulations (simulation results)
- ✅ algorithms (individual algorithm performance)
- ✅ simulation_ticks (tick-by-tick data)

---

## 🎯 Common Tasks in Compass

### **Task 1: View All Available Stocks**

1. Look at the Collections list
2. Count collections ending with `_historical`
3. Each represents one stock with data

### **Task 2: Check How Much Data You Have**

For each collection, Compass shows:
- **Document count** (number of records)
- **Storage size**
- **Indexes**

### **Task 3: View Latest Simulation Results**

1. Click **`simulations`** collection
2. Sort by `created_at` (descending)
3. Click on the first document
4. Expand `leaderboard` to see results
5. See `winner` for the best algorithm

### **Task 4: View Algorithm Code**

1. Click **`generations`** collection
2. Find your generation by `generation_id`
3. Expand the `algorithms` object
4. See the generated code for each model

### **Task 5: Analyze Trading Performance**

1. Click **`simulation_ticks`** collection
2. Filter by `simulation_id`
3. Sort by `tick_number`
4. See how portfolios changed over time

---

## 🔧 Troubleshooting

### **Can't Connect to MongoDB**

**Issue:** Connection refused or timeout

**Solution:**
```bash
# Check if MongoDB is running
mongosh

# If not running, start it:
brew services start mongodb-community  # Mac
sudo systemctl start mongod            # Linux
```

### **Database is Empty**

**Issue:** No collections visible

**Solution:**
```bash
# Fetch stock data first
cd backend
python fetch_minute_data_mongodb.py
```

### **Collections Not Showing**

**Issue:** Connected but can't see collections

**Solution:**
1. Make sure you're looking at the `ai_trader_battlefield` database
2. Click the database name in the sidebar
3. Refresh (Cmd/Ctrl + R)

---

## 📊 Visual Overview

When you open MongoDB Compass, you should see:

```
MongoDB Compass
├── Databases
│   └── ai_trader_battlefield ⭐ (YOUR DATABASE)
│       ├── Collections
│       │   ├── AAPL_historical (📊 daily data)
│       │   ├── AAPL_simulation (⏱️ minute data)
│       │   ├── AMZN_historical
│       │   ├── AMZN_simulation
│       │   ├── ... (more stocks)
│       │   ├── generations (🤖 algorithm requests)
│       │   ├── simulations (🎮 simulation results)
│       │   ├── algorithms (📝 code & performance)
│       │   └── simulation_ticks (📈 tick data)
```

---

## 🎉 Quick Start Checklist

- [ ] MongoDB is running locally
- [ ] MongoDB Compass is installed
- [ ] Connected to `mongodb://localhost:27017/ai_trader_battlefield`
- [ ] Can see `ai_trader_battlefield` database in sidebar
- [ ] Stock data fetched (run `fetch_minute_data_mongodb.py`)
- [ ] Collections are visible (e.g., `AAPL_historical`, `AAPL_simulation`)
- [ ] Can click and view documents

---

## 💡 Pro Tips

1. **Use Favorites:** Save your connection as a favorite for quick access
2. **Use Query History:** Compass saves your recent queries
3. **Use Schema View:** Quickly understand your data structure
4. **Export for Analysis:** Export to CSV for Excel/Python analysis
5. **Use Aggregations:** Build complex analytics pipelines visually

---

## 📞 Need Help?

**MongoDB Compass Docs:** https://docs.mongodb.com/compass/

**Connection String Format:**
```
mongodb://[host]:[port]/[database]

For AlgoClash:
mongodb://localhost:27017/ai_trader_battlefield
```

**Quick Test:**
```bash
# Test connection from command line
mongosh mongodb://localhost:27017/ai_trader_battlefield

# List collections
show collections

# Count documents
db.AAPL_historical.countDocuments()
```

---

## ✅ You're Ready!

Once you see your collections in MongoDB Compass, you can:
- ✅ Browse all stock data visually
- ✅ Filter and search documents
- ✅ View simulation results
- ✅ Analyze algorithm performance
- ✅ Export data for reporting
- ✅ Monitor your application in real-time

Happy exploring! 🎉

---

**Connection String:** `mongodb://localhost:27017/ai_trader_battlefield`
**Database:** `ai_trader_battlefield`
**Ready to use!** 🚀
