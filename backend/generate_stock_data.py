#!/usr/bin/env python3
"""
Script to fetch stock data from yfinance and save it as CSV for algorithm generation
"""

import yfinance as yf
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

def fetch_stock_data(symbol="AAPL", period_days=365, interval="1d"):
    """
    Fetch stock data from yfinance and return as DataFrame
    
    Args:
        symbol: Stock ticker symbol (default: AAPL)
        period_days: Number of days of historical data (default: 365)
        interval: Data interval - 1d, 1h, 1m, etc. (default: 1d)
        
    Returns:
        pandas.DataFrame: Stock data with OHLCV columns
    """
    print(f"📈 Fetching {symbol} data for past {period_days} days...")
    
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Calculate period string for yfinance
        if period_days <= 7:
            period = "7d"
        elif period_days <= 30:
            period = "1mo"
        elif period_days <= 90:
            period = "3mo" 
        elif period_days <= 180:
            period = "6mo"
        elif period_days <= 365:
            period = "1y"
        elif period_days <= 730:
            period = "2y"
        else:
            period = "5y"
        
        # Fetch the data
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            print(f"❌ No data retrieved for {symbol}")
            return None
            
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        
        # Add symbol column
        data['Symbol'] = symbol
        
        # Reorder columns to put Symbol first
        columns = ['Symbol', 'Date'] + [col for col in data.columns if col not in ['Symbol', 'Date']]
        data = data[columns]
        
        print(f"✅ Successfully fetched {len(data)} rows of data for {symbol}")
        print(f"📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return None

def save_stock_data_to_csv(data, file_path):
    """
    Save stock data DataFrame to CSV file
    
    Args:
        data: pandas.DataFrame with stock data
        file_path: Path where to save the CSV file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save to CSV
        data.to_csv(file_path, index=False)
        print(f"✅ Stock data saved to: {file_path}")
        
        # Print some sample data
        print(f"📊 Sample data (first 5 rows):")
        print(data.head())
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving data to {file_path}: {e}")
        return False

def generate_multiple_stocks_data(symbols=None, output_dir="data"):
    """
    Generate stock data for multiple symbols and save to CSV
    
    Args:
        symbols: List of stock symbols (default: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'])
        output_dir: Directory to save CSV files
    """
    if symbols is None:
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    print(f"🚀 Generating stock data for {len(symbols)} symbols...")
    
    all_data = []
    
    for symbol in symbols:
        data = fetch_stock_data(symbol, period_days=365, interval="1d")
        if data is not None:
            all_data.append(data)
    
    if all_data:
        # Combine all data into one DataFrame
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Save combined data
        combined_file = os.path.join(output_dir, "stock_data.csv")
        save_stock_data_to_csv(combined_data, combined_file)
        
        # Save individual files too
        for i, (symbol, data) in enumerate(zip(symbols, all_data)):
            individual_file = os.path.join(output_dir, f"{symbol}_data.csv")
            save_stock_data_to_csv(data, individual_file)
    
    print(f"🎉 Stock data generation completed!")

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

def generate_stock_data_for_ticker(ticker, output_dir="data"):
    """
    Generate stock data for a single ticker

    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'GME')
        output_dir: Directory to save CSV file

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n📊 Generating data for {ticker}...")

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_dir = os.path.join(script_dir, output_dir)

    # Fetch data
    data = fetch_stock_data(ticker, period_days=365, interval="1d")

    if data is not None:
        # Save individual file
        individual_file = os.path.join(full_output_dir, f"{ticker}_data.csv")
        return save_stock_data_to_csv(data, individual_file)

    return False

def main():
    """Main function to generate stock data"""
    print("📊 STOCK DATA GENERATOR")
    print("=" * 40)

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data")

    # Load tickers from stock_ticker.json
    symbols = load_tickers_from_json()
    print(f"📋 Generating data for: {', '.join(symbols)}")

    generate_multiple_stocks_data(symbols, output_dir)

if __name__ == "__main__":
    main()
