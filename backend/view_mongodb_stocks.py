#!/usr/bin/env python3
"""
View Stock Data in MongoDB
Simple script to view and query stock data stored in MongoDB
"""

from database import init_db, get_db
from datetime import datetime


def view_all_stocks():
    """View summary of all stocks in MongoDB"""
    print("\n" + "="*60)
    print("📊 STOCK DATA IN MONGODB")
    print("="*60)

    try:
        db = get_db()
        collection = db.stock_data

        # Get unique tickers
        tickers = collection.distinct('ticker')

        if not tickers:
            print("\n❌ No stock data found in MongoDB")
            print("\nRun this first:")
            print("   python fetch_and_store_stocks.py")
            return

        print(f"\n✅ Found {len(tickers)} stocks in database")
        print("\n" + "-"*60)

        for ticker in sorted(tickers):
            count = collection.count_documents({'ticker': ticker})

            # Get date range
            oldest = collection.find_one({'ticker': ticker}, sort=[('Date', 1)])
            newest = collection.find_one({'ticker': ticker}, sort=[('Date', -1)])

            print(f"\n{ticker}:")
            print(f"   📊 Records: {count}")
            if oldest and newest:
                print(f"   📅 Date range: {oldest.get('Date')} to {newest.get('Date')}")
                print(f"   💰 Latest close: ${newest.get('Close', 0):.2f}")
                print(f"   📈 Volume: {newest.get('Volume', 0):,}")

        print("\n" + "="*60)

    except Exception as e:
        print(f"❌ Error: {e}")


def view_stock_details(ticker):
    """View detailed data for a specific stock"""
    print(f"\n" + "="*60)
    print(f"📈 {ticker} DETAILED DATA")
    print("="*60)

    try:
        db = get_db()
        collection = db.stock_data

        # Get all records for this ticker
        records = list(collection.find({'ticker': ticker}).sort('Date', 1))

        if not records:
            print(f"\n❌ No data found for {ticker}")
            return

        print(f"\n✅ Found {len(records)} records for {ticker}")

        # Show first 5 and last 5 records
        print("\n📅 First 5 records:")
        print("-"*60)
        for i, record in enumerate(records[:5], 1):
            print(f"{i}. Date: {record.get('Date')}")
            print(f"   Open:   ${record.get('Open', 0):.2f}")
            print(f"   High:   ${record.get('High', 0):.2f}")
            print(f"   Low:    ${record.get('Low', 0):.2f}")
            print(f"   Close:  ${record.get('Close', 0):.2f}")
            print(f"   Volume: {record.get('Volume', 0):,}")

        if len(records) > 10:
            print("\n   ... ({} records in between) ...\n".format(len(records) - 10))

        print("\n📅 Last 5 records:")
        print("-"*60)
        for i, record in enumerate(records[-5:], len(records) - 4):
            print(f"{i}. Date: {record.get('Date')}")
            print(f"   Open:   ${record.get('Open', 0):.2f}")
            print(f"   High:   ${record.get('High', 0):.2f}")
            print(f"   Low:    ${record.get('Low', 0):.2f}")
            print(f"   Close:  ${record.get('Close', 0):.2f}")
            print(f"   Volume: {record.get('Volume', 0):,}")

        # Calculate statistics
        closes = [r.get('Close', 0) for r in records]
        volumes = [r.get('Volume', 0) for r in records]

        print("\n📊 Statistics:")
        print("-"*60)
        print(f"   Highest close: ${max(closes):.2f}")
        print(f"   Lowest close:  ${min(closes):.2f}")
        print(f"   Average close: ${sum(closes)/len(closes):.2f}")
        print(f"   Average volume: {sum(volumes)//len(volumes):,}")

        print("\n" + "="*60)

    except Exception as e:
        print(f"❌ Error: {e}")


def search_by_date_range(ticker, start_date, end_date):
    """Search stock data by date range"""
    print(f"\n" + "="*60)
    print(f"📅 {ticker} from {start_date} to {end_date}")
    print("="*60)

    try:
        db = get_db()
        collection = db.stock_data

        # Query by date range
        query = {
            'ticker': ticker,
            'Date': {'$gte': start_date, '$lte': end_date}
        }

        records = list(collection.find(query).sort('Date', 1))

        if not records:
            print(f"\n❌ No data found for {ticker} in this date range")
            return

        print(f"\n✅ Found {len(records)} records")
        print("\n" + "-"*60)

        for record in records:
            print(f"Date: {record.get('Date')} | "
                  f"Open: ${record.get('Open', 0):.2f} | "
                  f"Close: ${record.get('Close', 0):.2f} | "
                  f"Volume: {record.get('Volume', 0):,}")

        print("\n" + "="*60)

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function"""
    print("\n╔" + "="*58 + "╗")
    print("║" + " "*15 + "MongoDB Stock Data Viewer" + " "*19 + "║")
    print("╚" + "="*58 + "╝")

    # Initialize database
    try:
        init_db()
    except Exception as e:
        print(f"\n❌ Failed to connect to MongoDB: {e}")
        print("\nMake sure MongoDB is running:")
        print("  brew services start mongodb-community  # Mac")
        return

    while True:
        print("\nOptions:")
        print("1. View all stocks (summary)")
        print("2. View specific stock (detailed)")
        print("3. Search by date range")
        print("4. Exit")

        choice = input("\nSelect option (1-4): ")

        if choice == '1':
            view_all_stocks()

        elif choice == '2':
            ticker = input("Enter ticker symbol (e.g., AAPL): ").upper().strip()
            if ticker:
                view_stock_details(ticker)
            else:
                print("❌ Invalid ticker")

        elif choice == '3':
            ticker = input("Enter ticker symbol (e.g., AAPL): ").upper().strip()
            start = input("Enter start date (YYYY-MM-DD): ").strip()
            end = input("Enter end date (YYYY-MM-DD): ").strip()

            if ticker and start and end:
                search_by_date_range(ticker, start, end)
            else:
                print("❌ Invalid input")

        elif choice == '4':
            print("\n👋 Goodbye!\n")
            break

        else:
            print("❌ Invalid option")


if __name__ == "__main__":
    main()
