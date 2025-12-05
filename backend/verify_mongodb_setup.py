#!/usr/bin/env python3
"""
Verify MongoDB Setup for MongoDB Compass
Checks that everything is configured correctly and shows connection details
"""

from database import init_db, get_db
import sys


def verify_setup():
    """Verify MongoDB setup and configuration"""
    print("\n" + "="*60)
    print("🔍 VERIFYING MONGODB SETUP FOR COMPASS")
    print("="*60)

    # Step 1: Check connection
    print("\n📡 Step 1: Testing MongoDB Connection...")
    try:
        db = init_db()
        print(f"   ✅ Connected successfully!")
        print(f"   📊 Database name: {db.name}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print("\n   💡 Make sure MongoDB is running:")
        print("      brew services start mongodb-community  # Mac")
        print("      sudo systemctl start mongod            # Linux")
        return False

    # Step 2: Check connection string
    print("\n🔌 Step 2: Connection Details...")
    print("   Connection String for MongoDB Compass:")
    print("   ┌─────────────────────────────────────────────────────────┐")
    print("   │ mongodb://localhost:27017/ai_trader_battlefield         │")
    print("   └─────────────────────────────────────────────────────────┘")
    print(f"   Database: ai_trader_battlefield")
    print(f"   Host: localhost")
    print(f"   Port: 27017")

    # Step 3: Check collections
    print("\n📁 Step 3: Checking Collections...")
    collections = db.list_collection_names()

    if not collections:
        print("   ⚠️  No collections found (database is empty)")
        print("\n   💡 To populate with data, run:")
        print("      python fetch_minute_data_mongodb.py")
    else:
        print(f"   ✅ Found {len(collections)} collection(s):")

        # Separate stock collections from app collections
        stock_collections = []
        app_collections = []

        for coll in sorted(collections):
            if coll.endswith('_historical') or coll.endswith('_simulation'):
                stock_collections.append(coll)
            else:
                app_collections.append(coll)

        # Show stock collections
        if stock_collections:
            print("\n   📊 Stock Data Collections:")
            tickers = set()
            for coll in stock_collections:
                ticker = coll.replace('_historical', '').replace('_simulation', '')
                tickers.add(ticker)

            for ticker in sorted(tickers):
                hist_coll = f"{ticker}_historical"
                sim_coll = f"{ticker}_simulation"

                hist_count = db[hist_coll].count_documents({}) if hist_coll in collections else 0
                sim_count = db[sim_coll].count_documents({}) if sim_coll in collections else 0

                print(f"      {ticker}:")
                if hist_count > 0:
                    print(f"         ├── historical: {hist_count} days")
                if sim_count > 0:
                    print(f"         └── simulation: {sim_count} minutes")

        # Show app collections
        if app_collections:
            print("\n   🎮 Application Collections:")
            for coll in app_collections:
                count = db[coll].count_documents({})
                print(f"      ├── {coll}: {count} documents")

    # Step 4: MongoDB Compass Instructions
    print("\n" + "="*60)
    print("🎯 MONGODB COMPASS SETUP")
    print("="*60)

    print("\n1️⃣  Install MongoDB Compass:")
    print("   Download: https://www.mongodb.com/try/download/compass")
    print("   Or: brew install --cask mongodb-compass")

    print("\n2️⃣  Connect to Your Database:")
    print("   Open MongoDB Compass and paste this connection string:")
    print("")
    print("   ┌─────────────────────────────────────────────────────────┐")
    print("   │ mongodb://localhost:27017/ai_trader_battlefield         │")
    print("   └─────────────────────────────────────────────────────────┘")
    print("")
    print("   Then click 'Connect'")

    print("\n3️⃣  You Should See:")
    print("   Database: ai_trader_battlefield")
    if collections:
        print(f"   Collections: {len(collections)} total")
        if stock_collections:
            print(f"      - Stock data: {len(stock_collections)} collections")
        if app_collections:
            print(f"      - App data: {len(app_collections)} collections")
    else:
        print("   Collections: (empty - run fetch_minute_data_mongodb.py)")

    print("\n4️⃣  Browse Your Data:")
    print("   - Click on any collection to view documents")
    print("   - Use filters to search data")
    print("   - View schema, indexes, and statistics")

    # Step 5: Sample Queries
    if collections:
        print("\n" + "="*60)
        print("🔍 SAMPLE QUERIES FOR COMPASS")
        print("="*60)

        print("\nIn the Filter box, try these queries:")

        if any('_historical' in c for c in collections):
            print("\n📊 Historical Data Queries:")
            print('   {"ticker": "AAPL"}')
            print('   {"date": {$regex: "^2024-11"}}')
            print('   {"close": {$gt: 150}}')

        if any('_simulation' in c for c in collections):
            print("\n🎮 Simulation Data Queries:")
            print('   {"ticker": "AAPL"}')
            print('   {"datetime": {$regex: "09:30"}}')
            print('   {"volume": {$gt: 100000}}')

        if 'simulations' in collections:
            print("\n🏆 Simulation Results Queries:")
            print('   {"status": "completed"}')
            print('   {"stock_ticker": "AAPL"}')
            print('   {"leaderboard.roi": {$gt: 0.1}}')

    # Success
    print("\n" + "="*60)
    print("✅ VERIFICATION COMPLETE")
    print("="*60)

    if collections:
        print("\n🎉 Your MongoDB database is ready for MongoDB Compass!")
        print("\n📖 Full guide: MONGODB_COMPASS_GUIDE.md")
    else:
        print("\n💡 Next step: Fetch stock data")
        print("   python fetch_minute_data_mongodb.py")

    print("\n✅ Ready to connect with MongoDB Compass!")
    print("")

    return True


def main():
    """Main function"""
    print("\n╔" + "="*58 + "╗")
    print("║" + " "*12 + "MongoDB Compass Setup Checker" + " "*17 + "║")
    print("╚" + "="*58 + "╝")

    success = verify_setup()

    if success:
        print("\n💡 Quick Start:")
        print("   1. Open MongoDB Compass")
        print("   2. Paste: mongodb://localhost:27017/ai_trader_battlefield")
        print("   3. Click 'Connect'")
        print("   4. Browse your data!")
        print("")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
