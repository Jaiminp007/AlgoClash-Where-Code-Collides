# MongoDB Integration Setup Guide

This guide will help you set up MongoDB for AlgoClash.

## Prerequisites

1. **MongoDB installed and running**
   - Download from: https://www.mongodb.com/try/download/community
   - Or install via Homebrew (Mac): `brew install mongodb-community`
   - Or install via apt (Ubuntu): `sudo apt install mongodb`

2. **Python dependencies**
   - pymongo
   - python-dotenv

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install pymongo python-dotenv
```

### 2. Start MongoDB

**Mac (Homebrew):**
```bash
brew services start mongodb-community
```

**Ubuntu/Linux:**
```bash
sudo systemctl start mongod
```

**Windows:**
```bash
# MongoDB should start automatically after installation
# Or run: net start MongoDB
```

**Manual start:**
```bash
mongod --dbpath /path/to/data/directory
```

### 3. Verify MongoDB is Running

```bash
# Test connection
mongosh mongodb://localhost:27017

# Should see: Connected to MongoDB
# Type 'exit' to quit mongosh
```

### 4. Configure Environment

Update `backend/.env`:

```bash
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/ai_trader_battlefield
MONGODB_DATABASE=ai_trader_battlefield

# Existing config
OPENROUTER_API_KEY=your_api_key_here
```

### 5. Test MongoDB Integration

```bash
cd backend
python test_mongodb.py
```

Expected output:
```
╔==========================================================╗
║               MongoDB Test Suite                         ║
╚==========================================================╝

🧪 Testing MongoDB Connection
============================================================
✅ MongoDB connection successful!
📊 Database: ai_trader_battlefield
📦 Collections: ['generations', 'simulations', 'algorithms', 'simulation_ticks']

🧪 Testing Generation Creation
============================================================
✅ Generation created: test_gen_xxx
✅ Generation retrieved successfully
   Models: ['gpt-4', 'claude-3-haiku']
   Stock: AAPL_data.csv

...

📊 TEST SUMMARY
============================================================
✅ PASS: Connection Test
✅ PASS: Generation Create Test
✅ PASS: Simulation Create Test
✅ PASS: Update Operations Test
✅ PASS: Cleanup Test
============================================================
Results: 5/5 tests passed
============================================================

🎉 All tests passed! MongoDB is ready to use.
```

### 6. Update Flask App

**Option A: Use the new MongoDB version**

```bash
cd backend
mv app.py app_old.py
mv app_mongodb.py app.py
```

**Option B: Manually merge changes**

See the `app_mongodb.py` file for reference on what to change.

### 7. Start the Application

```bash
cd backend
python app.py
```

The application should now:
- ✅ Connect to MongoDB on startup
- ✅ Store all data in MongoDB collections
- ✅ Persist data across restarts

## Database Structure

### Collections

1. **generations**
   - Stores algorithm generation requests
   - Fields: generation_id, selected_models, selected_stock, status, algorithms, etc.

2. **simulations**
   - Stores simulation runs
   - Fields: simulation_id, generation_id, stock_ticker, results, leaderboard, etc.

3. **algorithms**
   - Stores individual algorithm code
   - Fields: simulation_id, generation_id, model_name, code, code_hash, performance_roi, etc.

4. **simulation_ticks**
   - Stores tick-by-tick simulation data for charts
   - Fields: simulation_id, tick_number, price, agent_portfolios, trades, etc.

### Indexes

The following indexes are automatically created for performance:

```javascript
// generations
db.generations.createIndex({ generation_id: 1 }, { unique: true })
db.generations.createIndex({ created_at: -1 })

// simulations
db.simulations.createIndex({ simulation_id: 1 }, { unique: true })
db.simulations.createIndex({ created_at: -1 })
db.simulations.createIndex({ status: 1 })

// algorithms
db.algorithms.createIndex({ simulation_id: 1, model_name: 1 })
db.algorithms.createIndex({ code_hash: 1 })

// simulation_ticks
db.simulation_ticks.createIndex({ simulation_id: 1, tick_number: 1 })
```

## MongoDB Shell Commands

### View Data

```bash
# Connect to MongoDB
mongosh mongodb://localhost:27017/ai_trader_battlefield

# List collections
show collections

# View generations
db.generations.find().pretty()

# View recent simulations
db.simulations.find().sort({created_at: -1}).limit(5).pretty()

# Count documents
db.generations.countDocuments()
db.simulations.countDocuments()

# Find by ID
db.generations.findOne({generation_id: "your-gen-id"})
```

### Cleanup

```bash
# Delete all test data
db.generations.deleteMany({generation_id: {$regex: '^test_'}})
db.simulations.deleteMany({simulation_id: {$regex: '^test_'}})

# Drop entire database (use with caution!)
db.dropDatabase()

# Delete old simulations (older than 7 days)
db.simulations.deleteMany({
  created_at: {
    $lt: new Date(Date.now() - 7*24*60*60*1000)
  }
})
```

## Troubleshooting

### MongoDB Connection Failed

**Error:** `Failed to connect to MongoDB`

**Solution:**
1. Check if MongoDB is running: `mongosh`
2. Verify connection string in `.env`
3. Check MongoDB logs: `tail -f /usr/local/var/log/mongodb/mongo.log` (Mac)

### Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Find process using port 27017
lsof -i :27017

# Kill the process
kill -9 <PID>

# Restart MongoDB
brew services restart mongodb-community
```

### Permission Denied

**Error:** `Permission denied`

**Solution:**
```bash
# Fix data directory permissions (Mac)
sudo chown -R `whoami` /usr/local/var/mongodb

# Linux
sudo chown -R mongodb:mongodb /var/lib/mongodb
```

### Index Creation Failed

**Error:** `Index creation failed`

**Solution:**
```bash
# Drop and recreate indexes
mongosh mongodb://localhost:27017/ai_trader_battlefield

db.generations.dropIndexes()
db.simulations.dropIndexes()
db.algorithms.dropIndexes()
db.simulation_ticks.dropIndexes()

# Restart application to recreate indexes
```

## API Changes

### Before (In-Memory)

```python
# Data stored in memory
running_simulations = {}
running_generations = {}

# Lost on restart
```

### After (MongoDB)

```python
# Data persisted in MongoDB
from database import GenerationRepository, SimulationRepository

gen_repo = GenerationRepository()
sim_repo = SimulationRepository()

# Survives restarts
generation = gen_repo.find_by_id(gen_id)
```

## Migration from In-Memory to MongoDB

If you have running simulations in the old version, they will be lost when switching to MongoDB. To preserve data:

1. **Before switching:**
   - Complete all running simulations
   - Or export important data

2. **After switching:**
   - All new data will be persisted
   - Data survives application restarts
   - Can query historical data

## Performance Tips

1. **Batch Tick Saves:**
   - Ticks are buffered and saved in batches of 10
   - Reduces database writes

2. **Index Usage:**
   - Queries use indexes for fast lookups
   - `generation_id` and `simulation_id` are indexed

3. **Cleanup Old Data:**
   - Regularly delete old simulations
   - Use MongoDB TTL indexes for automatic cleanup

## MongoDB Compass (GUI)

For a visual interface:

1. Download MongoDB Compass: https://www.mongodb.com/products/compass
2. Connect to: `mongodb://localhost:27017`
3. Browse `ai_trader_battlefield` database
4. View and query collections visually

## Next Steps

- ✅ MongoDB is configured
- ✅ Test script passes
- ✅ Application uses MongoDB

**Optional Enhancements:**
- Add user authentication
- Implement data export features
- Set up MongoDB Atlas (cloud hosting)
- Add backup automation
- Create admin dashboard

## Support

If you encounter issues:
1. Check MongoDB logs
2. Run test script: `python test_mongodb.py`
3. Verify MongoDB is running: `mongosh`
4. Check `.env` configuration

---

**Created:** $(date)
**Version:** 1.0
