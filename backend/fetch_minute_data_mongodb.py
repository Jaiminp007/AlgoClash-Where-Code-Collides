#!/usr/bin/env python3
"""
Fetch Minute-Level Stock Data to MongoDB
Organizes data into historical (for algo generation) and simulation (for trading day)

Data Structure:
- Historical data: All data BEFORE the simulation day (for training algorithms)
- Simulation data: Latest trading day data (for running the simulation)

Example for Nov 25, 2024 (Monday):
- Historical: All data from start_date to Nov 22, 2024 (Friday) - for algo generation
- Simulation: Nov 25, 2024 (Monday) minute data - for market simulation

Features:
- Option 1: Fetch all historical and simulation data
- Option 2: Fetch single stock
- Option 3: Update simulation day only
- Option 4: Incremental update - Only fetch missing data from last stored date to today
           (e.g., if last run was Nov 18, 2025, it will only fetch Nov 19-25, 2025)
"""

import requests
import json
from pathlib import Path
from datetime import datetime, timedelta
import time

from database import init_db, get_db


# ============================================================================
# CONFIGURATION
# ============================================================================
TWELVE_DATA_API_KEY = "ea26437639584f8fac81dd87583664aa"

# Stock tickers to fetch
STOCK_TICKERS = [
    "AAPL", "AMZN", "GME", "GOOGL", "META",
    "MSFT", "NFLX", "NVDA", "TSLA", "BAC", "ORCL"
]

# Date configuration
# For simulation: we want the PREVIOUS trading day (Monday, Nov 24, 2025)
# For historical: we want everything BEFORE that day
SIMULATION_DATE = "2025-12-01"  # Monday, November 24, 2025 (simulation day)
HISTORICAL_END_DATE = "2025-11-30"  # Friday, November 21, 2025 (day before simulation)
HISTORICAL_START_DATE = "2025-11-27"  # Start of historical data (January 1, 2025)


def get_last_trading_day():
    """
    Get the last trading day (skips weekends)
    Assumes today is the reference point
    """
    today = datetime.now()

    # If today is Monday, last trading day is Friday
    if today.weekday() == 0:  # Monday
        last_trading_day = today - timedelta(days=3)
    # If today is Sunday, last trading day is Friday
    elif today.weekday() == 6:  # Sunday
        last_trading_day = today - timedelta(days=2)
    # Otherwise, it's yesterday
    else:
        last_trading_day = today - timedelta(days=1)

    return last_trading_day.strftime("%Y-%m-%d")


def get_existing_dates_in_collection(collection):
    """
    Get all unique dates that already exist in a collection

    Args:
        collection: MongoDB collection

    Returns:
        Set of date strings (YYYY-MM-DD)
    """
    existing_dates = set()

    # Get all unique datetimes from the collection
    for doc in collection.find({}, {'datetime': 1}):
        if 'datetime' in doc:
            # Extract date from datetime string (e.g., "2025-11-24 09:30:00" -> "2025-11-24")
            date_str = doc['datetime'].split(' ')[0] if ' ' in doc['datetime'] else doc['datetime'][:10]
            existing_dates.add(date_str)

    return existing_dates


def get_last_date_in_collection(collection):
    """
    Get the last (most recent) date in a collection

    Args:
        collection: MongoDB collection

    Returns:
        Last date string (YYYY-MM-DD) or None if empty
    """
    # Find the document with the maximum datetime value
    result = collection.find_one(
        {},
        {'datetime': 1},
        sort=[('datetime', -1)]  # Sort descending to get latest
    )

    if result and 'datetime' in result:
        # Extract date from datetime string
        date_str = result['datetime'].split(' ')[0] if ' ' in result['datetime'] else result['datetime'][:10]
        return date_str

    return None


def get_missing_dates(start_date, end_date, existing_dates):
    """
    Get list of trading days that are missing from existing_dates

    Args:
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
        existing_dates: Set of existing date strings

    Returns:
        List of missing date strings in chronological order
    """
    missing_dates = []

    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    while current_date <= end_date_obj:
        # Skip weekends
        if current_date.weekday() < 5:  # Monday=0 to Friday=4
            date_str = current_date.strftime('%Y-%m-%d')
            if date_str not in existing_dates:
                missing_dates.append(date_str)

        current_date += timedelta(days=1)

    return missing_dates


def fetch_minute_data_from_api(symbol, date, api_key=TWELVE_DATA_API_KEY):
    """
    Fetch minute-level data for a specific date

    Args:
        symbol: Stock ticker
        date: Date in format 'YYYY-MM-DD'
        api_key: Twelve Data API key

    Returns:
        List of minute records or None
    """
    print(f"   📊 Fetching minute data for {symbol} on {date}...", end='', flush=True)

    try:
        base_url = "https://api.twelvedata.com/time_series"

        params = {
            'symbol': symbol,
            'interval': '1min',
            'outputsize': 390,  # Full trading day (6.5 hours * 60 min)
            'apikey': api_key,
            'format': 'JSON',
            'start_date': f"{date} 09:30:00",  # Market open
            'end_date': f"{date} 16:00:00"     # Market close
        }

        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()

        data_json = response.json()

        # Check for errors
        if 'status' in data_json and data_json['status'] == 'error':
            print(f" ❌ {data_json.get('message', 'Unknown error')}")
            return None

        values = data_json.get('values', [])

        if not values:
            print(f" ⚠️  No data")
            return None

        # Transform data
        records = []
        for item in values:
            record = {
                'datetime': item.get('datetime'),
                'open': float(item.get('open', 0)),
                'high': float(item.get('high', 0)),
                'low': float(item.get('low', 0)),
                'close': float(item.get('close', 0)),
                'volume': int(float(item.get('volume', 0)))
            }
            records.append(record)

        # Sort by datetime
        records.sort(key=lambda x: x['datetime'])

        print(f" ✅ {len(records)} minutes")
        return records

    except Exception as e:
        print(f" ❌ Error: {e}")
        return None


def fetch_historical_data_from_api(symbol, start_date, end_date, existing_collection=None, api_key=TWELVE_DATA_API_KEY):
    """
    Fetch historical minute-level data (for algo generation)
    Only fetches dates that are missing from the collection (incremental update)

    Args:
        symbol: Stock ticker
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        existing_collection: MongoDB collection to check for existing data
        api_key: Twelve Data API key

    Returns:
        List of minute records or None
    """
    print(f"   📈 Fetching historical minute data for {symbol} ({start_date} to {end_date})...")

    try:
        # Check which dates already exist
        existing_dates = set()
        if existing_collection is not None:
            print(f"      🔍 Checking for existing data...", end='', flush=True)
            existing_dates = get_existing_dates_in_collection(existing_collection)
            print(f" Found {len(existing_dates)} days already in database")

        # Get missing dates
        missing_dates = get_missing_dates(start_date, end_date, existing_dates)

        if not missing_dates:
            print(f"      ✅ All data already exists! No fetching needed.")
            return []

        print(f"      📊 Need to fetch {len(missing_dates)} trading days (already have {len(existing_dates)} days)")

        all_records = []
        fetched_days = 0

        # Fetch only missing dates
        for i, date_str in enumerate(missing_dates, 1):
            print(f"      📊 Day {i}/{len(missing_dates)}: {date_str}...", end='', flush=True)

            # Fetch minute data for this day
            day_records = fetch_minute_data_from_api(symbol, date_str, api_key)

            if day_records:
                all_records.extend(day_records)
                print(f" ✅ {len(day_records)} minutes")
                fetched_days += 1
            else:
                print(f" ⚠️  No data")

            # Rate limiting between days
            if i < len(missing_dates):
                time.sleep(8)  # Wait 8 seconds between API calls

        if not all_records:
            print(f"   ℹ️  No new data to fetch")
            return []

        # Sort all records by datetime
        all_records.sort(key=lambda x: x['datetime'])

        print(f"   ✅ Fetched: {len(all_records)} new minutes across {fetched_days} trading days")
        return all_records

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def store_stock_data_mongodb(ticker, simulation_date=None, historical_end_date=None,
                              historical_start_date=None, clear_existing=True):
    """
    Fetch and store both historical and simulation data for a ticker

    Args:
        ticker: Stock ticker symbol
        simulation_date: Date for simulation day (YYYY-MM-DD)
        historical_end_date: End date for historical data
        historical_start_date: Start date for historical data
        clear_existing: Whether to clear existing data for this ticker
    """
    # Use defaults if not provided
    if simulation_date is None:
        simulation_date = SIMULATION_DATE
    if historical_end_date is None:
        historical_end_date = HISTORICAL_END_DATE
    if historical_start_date is None:
        historical_start_date = HISTORICAL_START_DATE

    print(f"\n{'='*60}")
    print(f"📊 Processing {ticker}")
    print(f"{'='*60}")
    print(f"   Simulation day: {simulation_date}")
    print(f"   Historical range: {historical_start_date} to {historical_end_date}")

    try:
        db = get_db()

        # Collections
        historical_collection = db[f"{ticker}_historical"]
        simulation_collection = db[f"{ticker}_simulation"]

        # Clear existing data if requested
        if clear_existing:
            hist_deleted = historical_collection.delete_many({}).deleted_count
            sim_deleted = simulation_collection.delete_many({}).deleted_count
            if hist_deleted > 0 or sim_deleted > 0:
                print(f"   🗑️  Cleared old data (hist: {hist_deleted}, sim: {sim_deleted})")

        # Fetch historical data (minute-level for all days)
        print(f"\n   📚 Fetching Historical Data (minute-level for algo generation):")
        historical_records = fetch_historical_data_from_api(
            ticker,
            historical_start_date,
            historical_end_date
        )

        if historical_records:
            # Add metadata
            for record in historical_records:
                record['ticker'] = ticker
                record['data_type'] = 'historical'
                record['fetched_at'] = datetime.utcnow()

            # Store in MongoDB
            result = historical_collection.insert_many(historical_records)
            print(f"   💾 Stored {len(result.inserted_ids)} historical minute records")
        else:
            print(f"   ⚠️  No historical data fetched")

        # Wait for API rate limit
        print(f"   ⏳ Waiting 8 seconds (API rate limit)...")
        time.sleep(8)

        # Fetch simulation data (minute-level for one day)
        print(f"\n   🎮 Fetching Simulation Data (trading day {simulation_date}):")
        simulation_records = fetch_minute_data_from_api(ticker, simulation_date)

        if simulation_records:
            # Add metadata
            for record in simulation_records:
                record['ticker'] = ticker
                record['data_type'] = 'simulation'
                record['simulation_date'] = simulation_date
                record['fetched_at'] = datetime.utcnow()

            # Store in MongoDB
            result = simulation_collection.insert_many(simulation_records)
            print(f"   💾 Stored {len(result.inserted_ids)} simulation minutes")
        else:
            print(f"   ⚠️  No simulation data fetched")

        # Create indexes (both collections now use datetime for minute data)
        historical_collection.create_index('datetime')
        historical_collection.create_index('ticker')
        simulation_collection.create_index('datetime')
        simulation_collection.create_index('ticker')

        print(f"\n   ✅ {ticker} complete!")
        return True

    except Exception as e:
        print(f"   ❌ Error processing {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return False


def fetch_all_stocks(tickers=None):
    """
    Fetch all stocks with historical and simulation data

    Args:
        tickers: List of ticker symbols (defaults to STOCK_TICKERS)
    """
    print("\n" + "="*60)
    print("📊 FETCH MINUTE DATA FOR ALL STOCKS")
    print("="*60)

    # Initialize MongoDB
    try:
        db = init_db()
        print(f"✅ Connected to database: {db.name}")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        return

    if tickers is None:
        tickers = STOCK_TICKERS

    print(f"\n📋 Stocks to fetch: {len(tickers)}")
    print(f"   {', '.join(tickers)}")

    print(f"\n📅 Configuration:")
    print(f"   Simulation day: {SIMULATION_DATE}")
    print(f"   Historical: {HISTORICAL_START_DATE} to {HISTORICAL_END_DATE}")

    response = input("\nContinue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled")
        return

    # Process each ticker
    successful = []
    failed = []

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}")
        print("-"*60)

        success = store_stock_data_mongodb(ticker)

        if success:
            successful.append(ticker)
        else:
            failed.append(ticker)

        # Wait between tickers
        if i < len(tickers):
            print(f"\n⏳ Waiting 10 seconds before next ticker...")
            time.sleep(10)

    # Summary
    print("\n" + "="*60)
    print("✅ FETCH COMPLETE")
    print("="*60)
    print(f"   Successful: {len(successful)}/{len(tickers)}")
    print(f"   Failed: {len(failed)}/{len(tickers)}")

    if successful:
        print(f"\n✅ Successfully processed:")
        for ticker in successful:
            print(f"   - {ticker}")

    if failed:
        print(f"\n❌ Failed:")
        for ticker in failed:
            print(f"   - {ticker}")

    print("\n💡 View data:")
    print("   python view_minute_data.py")

    print("\n📊 Collections created:")
    for ticker in successful:
        print(f"   - {ticker}_historical (daily data for algo generation)")
        print(f"   - {ticker}_simulation (minute data for trading)")


def incremental_update_all_stocks(tickers=None):
    """
    Incrementally update all stocks from last stored date to today
    Only fetches missing data since the last fetch

    Args:
        tickers: List of ticker symbols (defaults to STOCK_TICKERS)
    """
    print("\n" + "="*60)
    print("🔄 INCREMENTAL UPDATE - Fetch Missing Data")
    print("="*60)

    # Initialize MongoDB
    try:
        db = init_db()
        print(f"✅ Connected to database: {db.name}")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        return

    if tickers is None:
        tickers = STOCK_TICKERS

    print(f"\n📋 Stocks to update: {len(tickers)}")
    print(f"   {', '.join(tickers)}")

    # Get today's date
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n📅 Target date: {today}")

    print("\n📊 Checking each stock for missing data...")

    # Check each stock to see what needs updating
    stocks_to_update = []
    for ticker in tickers:
        historical_collection = db[f"{ticker}_historical"]

        # Get the last date in the collection
        last_date = get_last_date_in_collection(historical_collection)

        if last_date is None:
            print(f"   ⚠️  {ticker}: No data found - use option 1 to fetch initial data")
            continue

        # Calculate next day after last date
        last_date_obj = datetime.strptime(last_date, '%Y-%m-%d')
        next_day = (last_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')

        # Check if there's data to fetch
        if next_day > today:
            print(f"   ✅ {ticker}: Up to date (last: {last_date})")
        else:
            print(f"   📊 {ticker}: Last date {last_date} → Will fetch from {next_day} to {today}")
            stocks_to_update.append({
                'ticker': ticker,
                'start_date': next_day,
                'end_date': today,
                'last_date': last_date
            })

    if not stocks_to_update:
        print("\n✅ All stocks are up to date! No fetching needed.")
        return

    print(f"\n📋 Summary: {len(stocks_to_update)} stocks need updates")
    response = input("\nContinue with incremental update? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled")
        return

    # Process each stock that needs updating
    successful = []
    failed = []

    for i, stock_info in enumerate(stocks_to_update, 1):
        ticker = stock_info['ticker']
        start_date = stock_info['start_date']
        end_date = stock_info['end_date']

        print(f"\n{'='*60}")
        print(f"[{i}/{len(stocks_to_update)}] 📊 Updating {ticker}")
        print(f"{'='*60}")
        print(f"   Last stored: {stock_info['last_date']}")
        print(f"   Fetching: {start_date} to {end_date}")

        try:
            historical_collection = db[f"{ticker}_historical"]

            # Fetch missing historical data
            print(f"\n   📈 Fetching missing data...")
            historical_records = fetch_historical_data_from_api(
                ticker,
                start_date,
                end_date,
                existing_collection=historical_collection
            )

            if historical_records and len(historical_records) > 0:
                # Add metadata
                for record in historical_records:
                    record['ticker'] = ticker
                    record['data_type'] = 'historical'
                    record['fetched_at'] = datetime.utcnow()

                # Store in MongoDB
                result = historical_collection.insert_many(historical_records)
                print(f"   💾 Stored {len(result.inserted_ids)} new minute records")
                successful.append(ticker)
            else:
                print(f"   ℹ️  No new data available")
                successful.append(ticker)

            # Wait between tickers
            if i < len(stocks_to_update):
                print(f"\n   ⏳ Waiting 10 seconds before next ticker...")
                time.sleep(10)

        except Exception as e:
            print(f"   ❌ Error updating {ticker}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(ticker)

    # Summary
    print("\n" + "="*60)
    print("✅ INCREMENTAL UPDATE COMPLETE")
    print("="*60)
    print(f"   Successful: {len(successful)}/{len(stocks_to_update)}")
    print(f"   Failed: {len(failed)}/{len(stocks_to_update)}")

    if successful:
        print(f"\n✅ Successfully updated:")
        for ticker in successful:
            print(f"   - {ticker}")

    if failed:
        print(f"\n❌ Failed:")
        for ticker in failed:
            print(f"   - {ticker}")


def add_historical_data_range(start_date, end_date, tickers=None):
    """
    Add historical data for a specific date range WITHOUT deleting existing data.
    Uses incremental logic - only fetches dates that don't already exist.
    
    Args:
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'  
        tickers: List of tickers (defaults to STOCK_TICKERS)
    """
    print("\n" + "="*60)
    print("📊 ADD HISTORICAL DATA (Preserving Existing)")
    print("="*60)
    
    # Initialize MongoDB
    try:
        db = init_db()
        print(f"✅ Connected to database: {db.name}")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        return
    
    if tickers is None:
        tickers = STOCK_TICKERS
    
    print(f"\n📋 Stocks to update: {len(tickers)}")
    print(f"   {', '.join(tickers)}")
    print(f"\n📅 Date range to add: {start_date} to {end_date}")
    
    # Show what already exists for each ticker
    print("\n📊 Current data status:")
    for ticker in tickers:
        historical_collection = db[f"{ticker}_historical"]
        count = historical_collection.count_documents({})
        existing_dates = get_existing_dates_in_collection(historical_collection)
        
        if count > 0:
            last_date = get_last_date_in_collection(historical_collection)
            print(f"   {ticker}: {count} records, {len(existing_dates)} days (last: {last_date})")
        else:
            print(f"   {ticker}: No existing data")
    
    response = input("\nContinue adding data? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled")
        return
    
    # Process each ticker
    successful = []
    failed = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(tickers)}] 📊 Adding data for {ticker}")
        print(f"{'='*60}")
        
        try:
            historical_collection = db[f"{ticker}_historical"]
            
            # Get existing dates to avoid duplicates
            existing_dates = get_existing_dates_in_collection(historical_collection)
            print(f"   📂 Existing: {len(existing_dates)} days in database")
            
            # Get missing dates in the requested range
            missing_dates = get_missing_dates(start_date, end_date, existing_dates)
            
            if not missing_dates:
                print(f"   ✅ All dates in range already exist - skipping")
                successful.append(ticker)
                continue
            
            print(f"   📊 Missing: {len(missing_dates)} days to fetch")
            print(f"      Dates: {', '.join(missing_dates[:5])}{'...' if len(missing_dates) > 5 else ''}")
            
            # Fetch missing data
            all_records = []
            for j, date_str in enumerate(missing_dates, 1):
                print(f"      [{j}/{len(missing_dates)}] Fetching {date_str}...", end='', flush=True)
                
                day_records = fetch_minute_data_from_api(ticker, date_str)
                
                if day_records:
                    # Add metadata
                    for record in day_records:
                        record['ticker'] = ticker
                        record['data_type'] = 'historical'
                        record['fetched_at'] = datetime.utcnow()
                    
                    all_records.extend(day_records)
                else:
                    print(f" ⚠️ No data")
                
                # Rate limiting
                if j < len(missing_dates):
                    time.sleep(8)
            
            # Store all fetched records
            if all_records:
                result = historical_collection.insert_many(all_records)
                print(f"   💾 Stored {len(result.inserted_ids)} new minute records")
                successful.append(ticker)
            else:
                print(f"   ℹ️ No new data to store")
                successful.append(ticker)
            
            # Wait between tickers
            if i < len(tickers):
                print(f"\n   ⏳ Waiting 10 seconds before next ticker...")
                time.sleep(10)
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            failed.append(ticker)
    
    # Summary
    print("\n" + "="*60)
    print("✅ DATA ADDITION COMPLETE")
    print("="*60)
    print(f"   Successful: {len(successful)}/{len(tickers)}")
    print(f"   Failed: {len(failed)}/{len(tickers)}")
    
    # Show final counts
    print("\n📊 Final data counts:")
    for ticker in tickers:
        historical_collection = db[f"{ticker}_historical"]
        count = historical_collection.count_documents({})
        existing_dates = get_existing_dates_in_collection(historical_collection)
        last_date = get_last_date_in_collection(historical_collection)
        print(f"   {ticker}: {count} records, {len(existing_dates)} days (last: {last_date})")


def main():
    """Main function"""
    print("\n╔" + "="*58 + "╗")
    print("║" + " "*10 + "Fetch Minute-Level Stock Data" + " "*18 + "║")
    print("╚" + "="*58 + "╝")

    print("\nData Organization:")
    print("  📚 Historical: Daily data (for algorithm generation)")
    print("  🎮 Simulation: Minute data for one trading day")

    print("\nOptions:")
    print("1. Fetch ALL stocks (historical + simulation data) - CLEARS EXISTING")
    print("2. Fetch single stock - CLEARS EXISTING")
    print("3. Update simulation day only (all stocks)")
    print("4. Incremental update (fetch only missing data from last date to today)")
    print("5. Add historical data range (PRESERVES existing data) ⭐")
    print("6. Cancel")

    choice = input("\nSelect option (1-6): ")

    if choice == '1':
        fetch_all_stocks()

    elif choice == '2':
        ticker = input("Enter ticker (e.g., AAPL): ").upper().strip()
        if ticker:
            store_stock_data_mongodb(ticker)
        else:
            print("❌ Invalid ticker")

    elif choice == '3':
        print("\nThis will update simulation data for all stocks")
        sim_date = input(f"Simulation date (default: {SIMULATION_DATE}): ").strip()
        if not sim_date:
            sim_date = SIMULATION_DATE

        for ticker in STOCK_TICKERS:
            print(f"\n🔄 Updating {ticker}...")
            db = get_db()
            collection = db[f"{ticker}_simulation"]
            collection.delete_many({})

            records = fetch_minute_data_from_api(ticker, sim_date)
            if records:
                for record in records:
                    record['ticker'] = ticker
                    record['data_type'] = 'simulation'
                    record['simulation_date'] = sim_date
                    record['fetched_at'] = datetime.utcnow()
                collection.insert_many(records)
                print(f"   ✅ Updated")

            time.sleep(8)

    elif choice == '4':
        incremental_update_all_stocks()

    elif choice == '5':
        print("\n📅 Add Historical Data Range (Preserves Existing)")
        print("-" * 40)
        
        start_date = input(f"Start date (default: {HISTORICAL_START_DATE}): ").strip()
        if not start_date:
            start_date = HISTORICAL_START_DATE
        
        end_date = input(f"End date (default: {HISTORICAL_END_DATE}): ").strip()
        if not end_date:
            end_date = HISTORICAL_END_DATE
        
        print(f"\n📊 Will add data from {start_date} to {end_date}")
        
        # Ask for specific tickers or all
        ticker_input = input("Enter tickers (comma-separated) or 'all' for all stocks: ").strip()
        
        if ticker_input.lower() == 'all' or not ticker_input:
            tickers = STOCK_TICKERS
        else:
            tickers = [t.strip().upper() for t in ticker_input.split(',')]
        
        add_historical_data_range(start_date, end_date, tickers)

    else:
        print("❌ Cancelled")


if __name__ == "__main__":
    main()
