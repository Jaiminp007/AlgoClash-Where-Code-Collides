#!/usr/bin/env python3
"""
Script to fetch daily stock data from Twelve Data API and save it as CSV for algorithm generation
Generates daily OHLCV data for multiple stocks

CONFIGURATION:
--------------
To change the date range for data fetching, modify the DEFAULT_START_DATE and DEFAULT_END_DATE
variables below. Set them to None to fetch the latest available data.

Example date ranges:
- Last 1 year: 252 trading days
- Last 6 months: 126 trading days
- Last 3 months: 63 trading days
"""

import requests
import pandas as pd
import os
import json
from pathlib import Path
import time

# ============================================================================
# CONFIGURE DATE RANGE HERE
# ============================================================================
# Change these dates to fetch data from a specific date range
# Format: 'YYYY-MM-DD'
# Set to None to fetch the latest available data
DEFAULT_START_DATE = "2023-01-01"  # Fetch from 1+ year ago for historical data
DEFAULT_END_DATE = "2024-11-17"  # End date (should be before simulation start)
# ============================================================================

def fetch_stock_data_1min(symbol="AAPL", api_key="ea26437639584f8fac81dd87583664aa", target_ticks=252,
                         start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """
    Fetch daily interval stock data from Twelve Data API

    Args:
        symbol: Stock ticker symbol
        api_key: Twelve Data API key
        target_ticks: Number of days to fetch (default 252 for ~1 year of trading days)
        start_date: Optional start date in format 'YYYY-MM-DD'
        end_date: Optional end date in format 'YYYY-MM-DD'

    Returns:
        DataFrame with OHLCV data for daily intervals
    """
    date_info = ""
    if start_date or end_date:
        date_info = f" from {start_date or 'earliest'} to {end_date or 'latest'}"

    print(f"📈 Fetching {target_ticks} days of daily data for {symbol}{date_info}...")

    try:
        # Twelve Data API endpoint for time series
        base_url = "https://api.twelvedata.com/time_series"

        params = {
            'symbol': symbol,
            'interval': '1day',
            'outputsize': target_ticks,  # Request daily data points
            'apikey': api_key,
            'format': 'JSON'
        }

        # Add date range if specified
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date

        # Make API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data_json = response.json()

        # Check for API errors
        if 'status' in data_json and data_json['status'] == 'error':
            print(f"❌ API Error: {data_json.get('message', 'Unknown error')}")
            return None

        # Extract values
        values = data_json.get('values', [])

        if not values:
            print(f"❌ No data retrieved for {symbol}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(values)

        # Rename columns to match expected format
        df = df.rename(columns={
            'datetime': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        # Convert data types
        df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
        df['High'] = pd.to_numeric(df['High'], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)

        # Add symbol column
        df['Symbol'] = symbol

        # Reorder columns
        df = df[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Reverse to get chronological order (API returns newest first)
        df = df.iloc[::-1].reset_index(drop=True)

        # Ensure we have exactly target_ticks rows
        if len(df) > target_ticks:
            df = df.tail(target_ticks).reset_index(drop=True)

        print(f"✅ Successfully fetched {len(df)} rows for {symbol}")
        print(f"📅 Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")

        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error fetching data for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return None



def save_stock_data_to_csv(data, file_path):
    """Save stock data DataFrame to CSV file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path, index=False)
        print(f"✅ Stock data saved to: {file_path}")
        print(data.head())
        return True
    except Exception as e:
        print(f"❌ Error saving data to {file_path}: {e}")
        return False


def generate_multiple_stocks_data(symbols=None, output_dir="data", api_key="ea26437639584f8fac81dd87583664aa",
                                 target_ticks=252, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """Generate daily stock data for multiple symbols and save to CSV.

    Args:
        symbols: List of stock ticker symbols
        output_dir: Output directory for CSV files
        api_key: Twelve Data API key
        target_ticks: Number of trading days to fetch (default 252 for ~1 year)
        start_date: Optional start date in format 'YYYY-MM-DD'
        end_date: Optional end date in format 'YYYY-MM-DD'

    Notes:
        Fetches daily OHLCV data for each symbol.
        Rate limiting: Free tier allows 8 API calls/minute, so we add delays between requests.
    """
    if symbols is None:
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']

    print(f"🚀 Generating daily stock data for {len(symbols)} symbols ({target_ticks} days each)...")
    collected = []  # list of (symbol, data)

    for i, symbol in enumerate(symbols):
        # Add delay to avoid rate limiting (free tier: 8 calls/minute)
        if i > 0:
            print("⏳ Waiting 8 seconds to avoid rate limiting...")
            time.sleep(8)

        data = fetch_stock_data_1min(symbol, api_key=api_key, target_ticks=target_ticks,
                                     start_date=start_date, end_date=end_date)
        if data is not None:
            collected.append((symbol, data))
        else:
            print(f"⚠️ Skipping save for {symbol}: no data fetched")

    if collected:
        combined_data = pd.concat([d for _, d in collected], ignore_index=True)
        combined_file = os.path.join(output_dir, "stock_data.csv")
        save_stock_data_to_csv(combined_data, combined_file)

        for sym, df in collected:
            individual_file = os.path.join(output_dir, f"{sym}_data.csv")
            save_stock_data_to_csv(df, individual_file)

    print("🎉 Stock data generation completed!")


def generate_stock_data_for_ticker(symbol: str, output_dir: str = "data",
                                   api_key: str = "ea26437639584f8fac81dd87583664aa",
                                   target_ticks: int = 252,
                                   start_date: str = DEFAULT_START_DATE,
                                   end_date: str = DEFAULT_END_DATE) -> bool:
    """Fetch and save daily CSV data for a single ticker.

    Args:
        symbol: Stock ticker symbol e.g. 'AAPL'.
        output_dir: Directory (relative to backend/ by default) to write CSVs into.
        api_key: Twelve Data API key.
        target_ticks: Number of trading days to fetch (default 252 for ~1 year).
        start_date: Optional start date in format 'YYYY-MM-DD'.
        end_date: Optional end date in format 'YYYY-MM-DD'.

    Returns:
        True if the CSV was written successfully, otherwise False.
    """
    try:
        symbol = (symbol or "").upper().strip()
        if not symbol:
            print("❌ No symbol provided")
            return False

        # Resolve output path; default is backend/data
        backend_dir = Path(__file__).resolve().parent
        out_path = Path(output_dir)
        if not out_path.is_absolute():
            out_path = backend_dir / out_path
        out_path.mkdir(parents=True, exist_ok=True)

        # Fetch daily data
        data = fetch_stock_data_1min(symbol=symbol, api_key=api_key, target_ticks=target_ticks,
                                     start_date=start_date, end_date=end_date)
        if data is None or data.empty:
            print(f"❌ No data to save for {symbol}")
            return False

        # Ensure required columns exist for downstream CSVTickGenerator
        # Rename common variations just in case
        col_renames = {}
        for req in ["Open", "High", "Low", "Close", "Volume"]:
            for col in data.columns:
                if col.lower() == req.lower() and col != req:
                    col_renames[col] = req
        if col_renames:
            data = data.rename(columns=col_renames)

        # Persist per-ticker file: {SYMBOL}_data.csv
        target_file = out_path / f"{symbol}_data.csv"
        data.to_csv(target_file, index=False)
        print(f"✅ Saved {symbol} CSV → {target_file} ({len(data)} days)")
        return True

    except Exception as e:
        print(f"❌ Error generating CSV for {symbol}: {e}")
        return False


def load_tickers_from_json():
    """Load stock tickers from stock_ticker.json"""
    try:
        script_dir = Path(__file__).resolve().parent
        json_path = script_dir / "open_router" / "stock_ticker.json"

        if not json_path.exists():
            print(f"⚠️ stock_ticker.json not found at {json_path}")
            print("Using default tickers...")
            return ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tickers = data.get("Stock_Ticker", [])
        print(f"✅ Loaded {len(tickers)} tickers from stock_ticker.json")
        return tickers

    except Exception as e:
        print(f"❌ Error reading stock_ticker.json: {e}")
        print("Using default tickers...")
        return ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']


def main():
    """Main function to generate stock data

    To change the date range, modify DEFAULT_START_DATE and DEFAULT_END_DATE at the top of this file.
    Set them to None to fetch the latest available data.

    Examples:
        # Use the default dates defined at the top of the file
        python3 generate_stock_data.py

        # Or call with custom dates:
        # generate_multiple_stocks_data(symbols, output_dir, start_date='2024-11-06 09:30:00', end_date='2024-11-06 16:00:00')
    """
    print("📊 STOCK DATA GENERATOR")
    print("=" * 40)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data")

    symbols = load_tickers_from_json()
    print(f"📋 Generating data for: {', '.join(symbols)}")

    # Uses DEFAULT_START_DATE and DEFAULT_END_DATE by default
    if DEFAULT_START_DATE and DEFAULT_END_DATE:
        print(f"📅 Using date range: {DEFAULT_START_DATE} to {DEFAULT_END_DATE}")
    else:
        print(f"📅 Fetching latest available data")

    generate_multiple_stocks_data(symbols, output_dir)


if __name__ == "__main__":
    main()
