#!/usr/bin/env python3
"""
View Minute-Level Stock Data in MongoDB
Shows historical and simulation data organized by ticker
"""

from database import init_db, get_db
from datetime import datetime


def list_all_tickers():
    """List all tickers that have data in MongoDB"""
    print("\n" + "="*60)
    print("📊 AVAILABLE STOCK DATA")
    print("="*60)

    try:
        db = get_db()

        # Get all collections
        collections = db.list_collection_names()

        # Extract unique tickers
        tickers = set()
        for coll in collections:
            if coll.endswith('_historical') or coll.endswith('_simulation'):
                ticker = coll.replace('_historical', '').replace('_simulation', '')
                tickers.add(ticker)

        if not tickers:
            print("\n❌ No stock data found")
            print("\nRun this first:")
            print("   python fetch_minute_data_mongodb.py")
            return []

        tickers = sorted(tickers)
        print(f"\n✅ Found {len(tickers)} stocks with data:\n")

        for ticker in tickers:
            hist_coll = f"{ticker}_historical"
            sim_coll = f"{ticker}_simulation"

            hist_count = db[hist_coll].count_documents({}) if hist_coll in collections else 0
            sim_count = db[sim_coll].count_documents({}) if sim_coll in collections else 0

            print(f"   {ticker}:")
            print(f"      📚 Historical: {hist_count} minutes")
            print(f"      🎮 Simulation: {sim_count} minutes")

        print("\n" + "="*60)
        return tickers

    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def view_historical_data(ticker):
    """View historical data for a ticker"""
    print(f"\n" + "="*60)
    print(f"📚 {ticker} - HISTORICAL DATA (minute-level for algo generation)")
    print("="*60)

    try:
        db = get_db()
        collection = db[f"{ticker}_historical"]

        count = collection.count_documents({})

        if count == 0:
            print(f"\n❌ No historical data for {ticker}")
            return

        # Get datetime range
        oldest = collection.find_one({}, sort=[('datetime', 1)])
        newest = collection.find_one({}, sort=[('datetime', -1)])

        # Extract date from datetime
        oldest_date = oldest['datetime'].split(' ')[0] if ' ' in oldest['datetime'] else oldest['datetime'][:10]
        newest_date = newest['datetime'].split(' ')[0] if ' ' in newest['datetime'] else newest['datetime'][:10]

        print(f"\n✅ Found {count} minutes of historical data")
        print(f"   📅 Date range: {oldest_date} to {newest_date}")
        print(f"   ⏰ Time range: {oldest['datetime']} to {newest['datetime']}")

        # Show first 10 minutes and last 10 minutes
        print(f"\n⏰ First 10 minutes:")
        print("-"*60)
        for record in collection.find().sort('datetime', 1).limit(10):
            print(f"   {record['datetime']} | O: ${record['open']:.2f} | "
                  f"H: ${record['high']:.2f} | L: ${record['low']:.2f} | "
                  f"C: ${record['close']:.2f} | V: {record['volume']:,}")

        if count > 20:
            print(f"\n   ... ({count - 20} minutes in between) ...")

        print(f"\n⏰ Last 10 minutes:")
        print("-"*60)
        for record in collection.find().sort('datetime', -1).limit(10):
            print(f"   {record['datetime']} | O: ${record['open']:.2f} | "
                  f"H: ${record['high']:.2f} | L: ${record['low']:.2f} | "
                  f"C: ${record['close']:.2f} | V: {record['volume']:,}")

        print("\n" + "="*60)

    except Exception as e:
        print(f"❌ Error: {e}")


def view_simulation_data(ticker):
    """View simulation data for a ticker"""
    print(f"\n" + "="*60)
    print(f"🎮 {ticker} - SIMULATION DATA (minute-by-minute)")
    print("="*60)

    try:
        db = get_db()
        collection = db[f"{ticker}_simulation"]

        count = collection.count_documents({})

        if count == 0:
            print(f"\n❌ No simulation data for {ticker}")
            return

        # Get time range
        first = collection.find_one({}, sort=[('datetime', 1)])
        last = collection.find_one({}, sort=[('datetime', -1)])

        sim_date = first.get('simulation_date', 'Unknown')

        print(f"\n✅ Found {count} minutes of simulation data")
        print(f"   📅 Trading day: {sim_date}")
        print(f"   ⏰ Time range: {first['datetime']} to {last['datetime']}")

        # Show opening minutes
        print(f"\n🔔 Opening minutes (9:30 AM):")
        print("-"*60)
        for record in collection.find().sort('datetime', 1).limit(10):
            print(f"   {record['datetime']} | O: ${record['open']:.2f} | "
                  f"H: ${record['high']:.2f} | L: ${record['low']:.2f} | "
                  f"C: ${record['close']:.2f} | V: {record['volume']:,}")

        if count > 20:
            print(f"\n   ... ({count - 20} minutes in between) ...")

        # Show closing minutes
        print(f"\n🔕 Closing minutes (4:00 PM):")
        print("-"*60)
        for record in collection.find().sort('datetime', -1).limit(10):
            print(f"   {record['datetime']} | O: ${record['open']:.2f} | "
                  f"H: ${record['high']:.2f} | L: ${record['low']:.2f} | "
                  f"C: ${record['close']:.2f} | V: {record['volume']:,}")

        # Calculate statistics
        all_records = list(collection.find().sort('datetime', 1))
        opens = [r['open'] for r in all_records]
        highs = [r['high'] for r in all_records]
        lows = [r['low'] for r in all_records]
        closes = [r['close'] for r in all_records]
        volumes = [r['volume'] for r in all_records]

        print(f"\n📊 Day Statistics:")
        print("-"*60)
        print(f"   Opening price: ${opens[0]:.2f}")
        print(f"   Closing price: ${closes[-1]:.2f}")
        print(f"   Daily change: ${closes[-1] - opens[0]:.2f} "
              f"({((closes[-1] / opens[0]) - 1) * 100:+.2f}%)")
        print(f"   Day high: ${max(highs):.2f}")
        print(f"   Day low: ${min(lows):.2f}")
        print(f"   Average volume: {sum(volumes) // len(volumes):,}")

        print("\n" + "="*60)

    except Exception as e:
        print(f"❌ Error: {e}")


def compare_data_split(ticker):
    """Show the data split between historical and simulation"""
    print(f"\n" + "="*60)
    print(f"📊 {ticker} - DATA ORGANIZATION")
    print("="*60)

    try:
        db = get_db()

        hist_coll = db[f"{ticker}_historical"]
        sim_coll = db[f"{ticker}_simulation"]

        hist_count = hist_coll.count_documents({})
        sim_count = sim_coll.count_documents({})

        print(f"\n📚 Historical Data (for Algorithm Generation):")
        print("-"*60)
        if hist_count > 0:
            oldest = hist_coll.find_one({}, sort=[('datetime', 1)])
            newest = hist_coll.find_one({}, sort=[('datetime', -1)])

            # Extract dates from datetime strings
            oldest_date = oldest['datetime'].split(' ')[0] if ' ' in oldest['datetime'] else oldest['datetime'][:10]
            newest_date = newest['datetime'].split(' ')[0] if ' ' in newest['datetime'] else newest['datetime'][:10]

            print(f"   Purpose: Train algorithms on past minute-level data")
            print(f"   Records: {hist_count} minutes")
            print(f"   Date Range: {oldest_date} to {newest_date}")
            print(f"   Time Range: {oldest['datetime']} to {newest['datetime']}")
            print(f"   Granularity: 1-minute (OHLCV)")
        else:
            print(f"   ❌ No historical data")

        print(f"\n🎮 Simulation Data (for Market Simulation):")
        print("-"*60)
        if sim_count > 0:
            first = sim_coll.find_one({}, sort=[('datetime', 1)])
            last = sim_coll.find_one({}, sort=[('datetime', -1)])
            sim_date = first.get('simulation_date', 'Unknown')
            print(f"   Purpose: Run algorithms in simulated trading day")
            print(f"   Records: {sim_count} minutes")
            print(f"   Trading day: {sim_date}")
            print(f"   Time: {first['datetime']} to {last['datetime']}")
            print(f"   Granularity: 1-minute (OHLCV)")
        else:
            print(f"   ❌ No simulation data")

        print(f"\n📋 Usage:")
        print("-"*60)
        print(f"   1. Generate algorithms using minute-level historical data")
        print(f"   2. Run simulation on minute-by-minute data")
        print(f"   3. Algorithms learn from granular price patterns")

        print("\n" + "="*60)

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function"""
    print("\n╔" + "="*58 + "╗")
    print("║" + " "*12 + "MongoDB Minute Data Viewer" + " "*20 + "║")
    print("╚" + "="*58 + "╝")

    # Initialize database
    try:
        init_db()
    except Exception as e:
        print(f"\n❌ Failed to connect to MongoDB: {e}")
        return

    while True:
        print("\nOptions:")
        print("1. List all tickers")
        print("2. View historical data (specific ticker)")
        print("3. View simulation data (specific ticker)")
        print("4. View data organization (specific ticker)")
        print("5. Exit")

        choice = input("\nSelect option (1-5): ")

        if choice == '1':
            list_all_tickers()

        elif choice == '2':
            ticker = input("Enter ticker (e.g., AAPL): ").upper().strip()
            if ticker:
                view_historical_data(ticker)
            else:
                print("❌ Invalid ticker")

        elif choice == '3':
            ticker = input("Enter ticker (e.g., AAPL): ").upper().strip()
            if ticker:
                view_simulation_data(ticker)
            else:
                print("❌ Invalid ticker")

        elif choice == '4':
            ticker = input("Enter ticker (e.g., AAPL): ").upper().strip()
            if ticker:
                compare_data_split(ticker)
            else:
                print("❌ Invalid ticker")

        elif choice == '5':
            print("\n👋 Goodbye!\n")
            break

        else:
            print("❌ Invalid option")


if __name__ == "__main__":
    main()
