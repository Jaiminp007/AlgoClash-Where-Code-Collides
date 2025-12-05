"""
Store Stock Data in MongoDB
Reads CSV files from backend/data/ and stores them in MongoDB for easy access
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from database import init_db, get_db


def store_stock_data():
    """Read stock CSV files and store in MongoDB"""
    print("\n" + "="*60)
    print("📊 STORING STOCK DATA IN MONGODB")
    print("="*60)

    try:
        # Initialize database
        db = init_db()
        collection = db.stock_data

        # Get data directory
        data_dir = Path(__file__).parent / "data"

        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            return

        # Find all CSV files
        csv_files = list(data_dir.glob("*_data.csv"))

        if not csv_files:
            print(f"❌ No CSV files found in {data_dir}")
            return

        print(f"\n📁 Found {len(csv_files)} stock data files:")
        for f in csv_files:
            print(f"   - {f.name}")

        # Ask for confirmation
        print(f"\n⚠️  This will store {len(csv_files)} stocks in MongoDB")
        response = input("Continue? (yes/no): ")

        if response.lower() not in ['yes', 'y']:
            print("❌ Operation cancelled")
            return

        # Process each CSV file
        total_records = 0
        stocks_processed = []

        for csv_file in csv_files:
            ticker = csv_file.stem.replace('_data', '').upper()
            print(f"\n📈 Processing {ticker}...")

            try:
                # Read CSV
                df = pd.read_csv(csv_file)

                # Convert DataFrame to list of dictionaries
                records = df.to_dict('records')

                # Add metadata to each record
                for record in records:
                    record['ticker'] = ticker
                    record['imported_at'] = datetime.utcnow()

                    # Convert string dates to datetime if needed
                    if 'Date' in record and isinstance(record['Date'], str):
                        try:
                            record['date_obj'] = pd.to_datetime(record['Date'])
                        except:
                            pass

                # Delete existing data for this ticker
                delete_result = collection.delete_many({'ticker': ticker})
                if delete_result.deleted_count > 0:
                    print(f"   🗑️  Removed {delete_result.deleted_count} old records")

                # Insert new data
                if records:
                    insert_result = collection.insert_many(records)
                    count = len(insert_result.inserted_ids)
                    total_records += count
                    stocks_processed.append(ticker)
                    print(f"   ✅ Inserted {count} records for {ticker}")
                else:
                    print(f"   ⚠️  No records found in {csv_file.name}")

            except Exception as e:
                print(f"   ❌ Error processing {ticker}: {e}")

        # Create indexes for better query performance
        print("\n🔧 Creating indexes...")
        try:
            collection.create_index('ticker')
            collection.create_index([('ticker', 1), ('Date', 1)])
            collection.create_index('date_obj')
            print("   ✅ Indexes created")
        except Exception as e:
            print(f"   ⚠️  Index creation warning: {e}")

        # Summary
        print("\n" + "="*60)
        print("✅ STOCK DATA STORAGE COMPLETE")
        print("="*60)
        print(f"   Stocks processed: {len(stocks_processed)}")
        print(f"   Total records: {total_records}")
        print(f"   Average records per stock: {total_records // len(stocks_processed) if stocks_processed else 0}")

        print("\n📊 Stocks stored:")
        for ticker in sorted(stocks_processed):
            count = collection.count_documents({'ticker': ticker})
            print(f"   - {ticker}: {count} records")

        print("\n💡 Query examples:")
        print("   mongosh mongodb://localhost:27017/ai_trader_battlefield")
        print("   db.stock_data.find({ticker: 'AAPL'}).limit(5)")
        print("   db.stock_data.countDocuments({ticker: 'AAPL'})")
        print("   db.stock_data.distinct('ticker')")

    except Exception as e:
        print(f"\n❌ Error storing stock data: {e}")
        import traceback
        traceback.print_exc()


def view_stock_data():
    """View stock data stored in MongoDB"""
    print("\n" + "="*60)
    print("👀 VIEWING STOCK DATA FROM MONGODB")
    print("="*60)

    try:
        # Initialize database
        db = init_db()
        collection = db.stock_data

        # Get unique tickers
        tickers = collection.distinct('ticker')

        if not tickers:
            print("ℹ️  No stock data found in MongoDB")
            return

        print(f"\n📊 Available stocks: {', '.join(sorted(tickers))}")

        # Show summary for each ticker
        print("\n" + "-"*60)
        for ticker in sorted(tickers):
            count = collection.count_documents({'ticker': ticker})
            sample = collection.find_one({'ticker': ticker})

            print(f"\n{ticker}:")
            print(f"   Records: {count}")

            if sample:
                print(f"   Sample date: {sample.get('Date', 'N/A')}")
                print(f"   Open: ${sample.get('Open', 'N/A')}")
                print(f"   Close: ${sample.get('Close', 'N/A')}")
                print(f"   Volume: {sample.get('Volume', 'N/A')}")

        print("\n" + "="*60)

    except Exception as e:
        print(f"\n❌ Error viewing stock data: {e}")
        import traceback
        traceback.print_exc()


def export_stock_data():
    """Export stock data from MongoDB back to CSV"""
    print("\n" + "="*60)
    print("💾 EXPORTING STOCK DATA FROM MONGODB")
    print("="*60)

    try:
        # Initialize database
        db = init_db()
        collection = db.stock_data

        # Get unique tickers
        tickers = collection.distinct('ticker')

        if not tickers:
            print("ℹ️  No stock data found in MongoDB")
            return

        print(f"\n📊 Found {len(tickers)} stocks in MongoDB")

        # Create export directory
        export_dir = Path(__file__).parent / "data_export"
        export_dir.mkdir(exist_ok=True)

        # Export each ticker
        for ticker in sorted(tickers):
            print(f"\n📤 Exporting {ticker}...")

            # Get all records for this ticker
            records = list(collection.find({'ticker': ticker}, {'_id': 0, 'ticker': 0, 'imported_at': 0}))

            if records:
                # Convert to DataFrame
                df = pd.DataFrame(records)

                # Remove date_obj if it exists (keep original Date string)
                if 'date_obj' in df.columns:
                    df = df.drop('date_obj', axis=1)

                # Export to CSV
                output_file = export_dir / f"{ticker}_data.csv"
                df.to_csv(output_file, index=False)

                print(f"   ✅ Exported {len(records)} records to {output_file}")

        print("\n" + "="*60)
        print(f"✅ Export complete: {export_dir}")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error exporting stock data: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    print("\n╔" + "="*58 + "╗")
    print("║" + " "*15 + "Stock Data Manager" + " "*25 + "║")
    print("╚" + "="*58 + "╝")

    print("\nOptions:")
    print("1. Store stock data from CSV files to MongoDB")
    print("2. View stock data in MongoDB")
    print("3. Export stock data from MongoDB to CSV")
    print("4. Cancel")

    choice = input("\nSelect option (1-4): ")

    if choice == '1':
        store_stock_data()
    elif choice == '2':
        view_stock_data()
    elif choice == '3':
        export_stock_data()
    else:
        print("❌ Operation cancelled")


if __name__ == "__main__":
    main()
