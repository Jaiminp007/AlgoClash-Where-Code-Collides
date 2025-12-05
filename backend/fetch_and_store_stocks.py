#!/usr/bin/env python3
"""
Fetch Stock Data Directly to MongoDB
Fetches stock data from Twelve Data API and stores directly in MongoDB (no CSV files)
"""

import requests
import json
from pathlib import Path
from datetime import datetime
import time

from database import init_db, get_db


# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2024-11-17"
TWELVE_DATA_API_KEY = "ea26437639584f8fac81dd87583664aa"

# Stock tickers to fetch
STOCK_TICKERS = [
    "AAPL", "AMZN", "GME", "GOOGL", "META",
    "MSFT", "NFLX", "NVDA", "TSLA", "BAC", "ORCL"
]


def fetch_stock_data_from_api(symbol, api_key=TWELVE_DATA_API_KEY, target_ticks=252,
                               start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """
    Fetch daily stock data from Twelve Data API

    Args:
        symbol: Stock ticker symbol
        api_key: Twelve Data API key
        target_ticks: Number of days to fetch
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'

    Returns:
        List of data records or None if error
    """
    date_info = f" from {start_date or 'earliest'} to {end_date or 'latest'}"
    print(f"   📈 Fetching {target_ticks} days for {symbol}{date_info}...", end='')

    try:
        base_url = "https://api.twelvedata.com/time_series"

        params = {
            'symbol': symbol,
            'interval': '1day',
            'outputsize': target_ticks,
            'apikey': api_key,
            'format': 'JSON'
        }

        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date

        # Make API request
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()

        data_json = response.json()

        # Check for API errors
        if 'status' in data_json and data_json['status'] == 'error':
            print(f" ❌ API Error: {data_json.get('message', 'Unknown error')}")
            return None

        # Extract values
        values = data_json.get('values', [])

        if not values:
            print(f" ❌ No data")
            return None

        # Transform data
        records = []
        for item in values:
            record = {
                'Date': item.get('datetime'),
                'Open': float(item.get('open', 0)),
                'High': float(item.get('high', 0)),
                'Low': float(item.get('low', 0)),
                'Close': float(item.get('close', 0)),
                'Volume': int(float(item.get('volume', 0))),
                'ticker': symbol,
                'fetched_at': datetime.utcnow()
            }
            records.append(record)

        # Sort by date (oldest first)
        records.sort(key=lambda x: x['Date'])

        print(f" ✅ {len(records)} records")
        return records

    except requests.exceptions.RequestException as e:
        print(f" ❌ Request failed: {e}")
        return None
    except Exception as e:
        print(f" ❌ Error: {e}")
        return None


def store_stocks_in_mongodb(tickers=None, clear_existing=True):
    """
    Fetch stocks from API and store directly in MongoDB

    Args:
        tickers: List of ticker symbols (defaults to STOCK_TICKERS)
        clear_existing: Whether to clear existing data first
    """
    print("\n" + "="*60)
    print("📊 FETCH & STORE STOCKS DIRECTLY TO MONGODB")
    print("="*60)

    # Initialize MongoDB
    try:
        db = init_db()
        collection = db.stock_data
        print(f"✅ Connected to database: {db.name}")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        print("\nMake sure MongoDB is running:")
        print("  brew services start mongodb-community  # Mac")
        return

    # Use default tickers if none provided
    if tickers is None:
        tickers = STOCK_TICKERS

    print(f"\n📋 Will fetch {len(tickers)} stocks: {', '.join(tickers)}")
    print(f"📅 Date range: {DEFAULT_START_DATE} to {DEFAULT_END_DATE}")

    # Clear existing data if requested
    if clear_existing:
        print("\n🗑️  Clearing existing stock data...")
        result = collection.delete_many({})
        if result.deleted_count > 0:
            print(f"   ✅ Deleted {result.deleted_count} old records")
        else:
            print(f"   ℹ️  No existing data to clear")

    # Fetch and store each stock
    print("\n📡 Fetching data from API...")
    print("-"*60)

    total_records = 0
    successful_tickers = []
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}")

        # Fetch from API
        records = fetch_stock_data_from_api(ticker)

        if records:
            try:
                # Insert into MongoDB
                result = collection.insert_many(records)
                count = len(result.inserted_ids)
                total_records += count
                successful_tickers.append(ticker)
                print(f"   💾 Stored {count} records in MongoDB")

            except Exception as e:
                print(f"   ❌ Failed to store in MongoDB: {e}")
                failed_tickers.append(ticker)
        else:
            failed_tickers.append(ticker)

        # Rate limiting (API allows 8 requests/minute for free tier)
        if i < len(tickers):
            print(f"   ⏳ Waiting 8 seconds (API rate limit)...")
            time.sleep(8)

    # Create indexes
    print("\n🔧 Creating indexes...")
    try:
        collection.create_index('ticker')
        collection.create_index([('ticker', 1), ('Date', 1)])
        collection.create_index('Date')
        print("   ✅ Indexes created")
    except Exception as e:
        print(f"   ⚠️  Index warning: {e}")

    # Summary
    print("\n" + "="*60)
    print("✅ FETCH & STORE COMPLETE")
    print("="*60)
    print(f"   Successful: {len(successful_tickers)}/{len(tickers)} stocks")
    print(f"   Total records: {total_records}")
    if successful_tickers:
        print(f"   Average per stock: {total_records // len(successful_tickers)}")

    if successful_tickers:
        print(f"\n✅ Successfully stored:")
        for ticker in sorted(successful_tickers):
            count = collection.count_documents({'ticker': ticker})
            print(f"   - {ticker}: {count} records")

    if failed_tickers:
        print(f"\n❌ Failed to fetch:")
        for ticker in failed_tickers:
            print(f"   - {ticker}")

    # Show how to view data
    print("\n💡 View your data:")
    print("="*60)
    print("\n1️⃣  MongoDB Shell:")
    print("   mongosh mongodb://localhost:27017/ai_trader_battlefield")
    print("   > db.stock_data.distinct('ticker')")
    print("   > db.stock_data.find({ticker: 'AAPL'}).limit(5).pretty()")
    print("   > db.stock_data.countDocuments({ticker: 'AAPL'})")

    print("\n2️⃣  Python:")
    print("   from database import get_db")
    print("   db = get_db()")
    print("   data = list(db.stock_data.find({'ticker': 'AAPL'}).limit(5))")

    print("\n3️⃣  View script:")
    print("   python view_mongodb_stocks.py")

    print("\n✅ Stock data is now in MongoDB!\n")


def update_single_stock(ticker):
    """Update a single stock (useful for refreshing data)"""
    print(f"\n🔄 Updating {ticker} in MongoDB...")

    try:
        db = get_db()
        collection = db.stock_data

        # Delete existing data for this ticker
        delete_result = collection.delete_many({'ticker': ticker})
        if delete_result.deleted_count > 0:
            print(f"   🗑️  Removed {delete_result.deleted_count} old records")

        # Fetch new data
        records = fetch_stock_data_from_api(ticker)

        if records:
            result = collection.insert_many(records)
            print(f"   ✅ Stored {len(result.inserted_ids)} new records")
            return True
        else:
            print(f"   ❌ Failed to fetch data")
            return False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Main function"""
    print("\n╔" + "="*58 + "╗")
    print("║" + " "*10 + "Fetch & Store Stocks in MongoDB" + " "*17 + "║")
    print("╚" + "="*58 + "╝")

    print("\nOptions:")
    print("1. Fetch ALL stocks and store in MongoDB (clears existing)")
    print("2. Fetch ALL stocks and ADD to existing data")
    print("3. Update a single stock")
    print("4. Custom stock list")
    print("5. Cancel")

    choice = input("\nSelect option (1-5): ")

    if choice == '1':
        store_stocks_in_mongodb(clear_existing=True)

    elif choice == '2':
        store_stocks_in_mongodb(clear_existing=False)

    elif choice == '3':
        ticker = input("Enter ticker symbol (e.g., AAPL): ").upper().strip()
        if ticker:
            update_single_stock(ticker)
        else:
            print("❌ Invalid ticker")

    elif choice == '4':
        tickers_input = input("Enter tickers separated by commas (e.g., AAPL,MSFT,GOOGL): ")
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        if tickers:
            store_stocks_in_mongodb(tickers=tickers, clear_existing=False)
        else:
            print("❌ No valid tickers entered")

    else:
        print("❌ Cancelled")


if __name__ == "__main__":
    main()
