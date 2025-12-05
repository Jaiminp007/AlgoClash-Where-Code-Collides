"""
MongoDB Quick Start Script
One-stop script to clear and populate MongoDB with stock data
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from database import init_db, get_db
import pandas as pd
from datetime import datetime


def main():
    """Quick start MongoDB setup"""
    print("\n╔" + "="*58 + "╗")
    print("║" + " "*12 + "MongoDB Quick Start for AlgoClash" + " "*13 + "║")
    print("╚" + "="*58 + "╝")

    print("\nThis script will:")
    print("  1. Clear all existing data in MongoDB")
    print("  2. Store stock data from CSV files")
    print("  3. Show you how to view the data")

    response = input("\nContinue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled")
        return

    # Step 1: Initialize MongoDB
    print("\n" + "="*60)
    print("📡 Step 1: Connecting to MongoDB")
    print("="*60)

    try:
        db = init_db()
        print(f"✅ Connected to database: {db.name}")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        print("\nMake sure MongoDB is running:")
        print("  brew services start mongodb-community  # Mac")
        print("  sudo systemctl start mongod            # Linux")
        return

    # Step 2: Clear existing data
    print("\n" + "="*60)
    print("🗑️  Step 2: Clearing existing data")
    print("="*60)

    try:
        collections = db.list_collection_names()
        total_deleted = 0

        for collection_name in collections:
            count = db[collection_name].count_documents({})
            if count > 0:
                result = db[collection_name].delete_many({})
                print(f"   ✅ Deleted {result.deleted_count} documents from '{collection_name}'")
                total_deleted += result.deleted_count

        if total_deleted == 0:
            print("   ℹ️  Database was already empty")
        else:
            print(f"   ✅ Total deleted: {total_deleted} documents")

    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        return

    # Step 3: Store stock data
    print("\n" + "="*60)
    print("📊 Step 3: Storing stock data from CSV files")
    print("="*60)

    try:
        # Get data directory
        data_dir = Path(__file__).parent / "data"

        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            print("   Run generate_stock_data.py first to create CSV files")
            return

        # Find CSV files
        csv_files = list(data_dir.glob("*_data.csv"))

        if not csv_files:
            print(f"❌ No stock CSV files found in {data_dir}")
            print("   Run generate_stock_data.py first to download stock data")
            return

        print(f"   Found {len(csv_files)} stock data files")

        # Create stock_data collection
        collection = db.stock_data

        total_records = 0
        stocks_stored = []

        for csv_file in csv_files:
            ticker = csv_file.stem.replace('_data', '').upper()
            print(f"   📈 Processing {ticker}...", end='')

            try:
                # Read CSV
                df = pd.read_csv(csv_file)

                # Convert to list of dictionaries
                records = df.to_dict('records')

                # Add metadata
                for record in records:
                    record['ticker'] = ticker
                    record['imported_at'] = datetime.utcnow()

                # Insert records
                if records:
                    collection.insert_many(records)
                    total_records += len(records)
                    stocks_stored.append(ticker)
                    print(f" ✅ {len(records)} records")
                else:
                    print(f" ⚠️  No data")

            except Exception as e:
                print(f" ❌ Error: {e}")

        # Create indexes
        print(f"\n   🔧 Creating indexes...")
        collection.create_index('ticker')
        collection.create_index([('ticker', 1), ('Date', 1)])
        print(f"   ✅ Indexes created")

        print(f"\n   ✅ Stored {len(stocks_stored)} stocks with {total_records} total records")

    except Exception as e:
        print(f"❌ Error storing stock data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Summary and next steps
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)

    print(f"\n📊 Stocks in MongoDB: {', '.join(sorted(stocks_stored))}")

    print("\n💡 How to view your data:")
    print("="*60)

    print("\n1️⃣  Using mongosh (MongoDB Shell):")
    print("   mongosh mongodb://localhost:27017/ai_trader_battlefield")
    print("   > db.stock_data.find({ticker: 'AAPL'}).limit(5).pretty()")
    print("   > db.stock_data.countDocuments({ticker: 'AAPL'})")
    print("   > db.stock_data.distinct('ticker')")

    print("\n2️⃣  Using Python:")
    print("   python")
    print("   >>> from database import get_db")
    print("   >>> db = get_db()")
    print("   >>> db.stock_data.find_one({'ticker': 'AAPL'})")

    print("\n3️⃣  Using MongoDB Compass (GUI):")
    print("   Download: https://www.mongodb.com/products/compass")
    print("   Connect to: mongodb://localhost:27017")
    print("   Browse: ai_trader_battlefield > stock_data")

    print("\n4️⃣  View in Terminal (run this script):")
    print("   python store_stock_data_mongodb.py")
    print("   Select option 2 to view data")

    print("\n🚀 Next steps:")
    print("="*60)
    print("1. View your data using one of the methods above")
    print("2. Run the Flask app: python app.py")
    print("3. Generate algorithms and run simulations")
    print("4. Check MongoDB for simulation results!")

    print("\n✅ MongoDB is ready to use!\n")


if __name__ == "__main__":
    main()
