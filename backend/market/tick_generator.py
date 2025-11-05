"""
Tick Generator for Market Simulation
Provides stock price data for trading simulation from CSV files or Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import threading
from typing import Iterator, Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import random


class TickData:
    """Represents a single market tick with price and volume data."""

    def __init__(self, timestamp: datetime, open_price: float, high: float,
                 low: float, close: float, volume: int, symbol: str):
        self.timestamp = timestamp
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.symbol = symbol

    def __repr__(self):
        return f"TickData({self.symbol}, {self.close:.2f}, {self.timestamp})"


class CSVTickGenerator:
    """
    Generates market ticks from pre-downloaded CSV files.
    Faster and works offline - no need for live yfinance downloads.
    """

    def __init__(self, symbol: str, data_dir: Optional[str] = None):
        """
        Initialize the CSV tick generator.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            data_dir: Directory containing CSV files (defaults to backend/data)
        """
        self.symbol = symbol.upper()
        self.data: Optional[pd.DataFrame] = None
        self.current_index = 0

        # Default to backend/data directory
        if data_dir is None:
            backend_dir = Path(__file__).resolve().parent.parent
            data_dir = backend_dir / "data"
        else:
            data_dir = Path(data_dir)

        self.csv_path = data_dir / f"{self.symbol}_data.csv"
        self._load_csv()

    def _load_csv(self):
        """Load data from CSV file."""
        try:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

            print(f"📈 Loading {self.symbol} data from CSV: {self.csv_path}")
            self.data = pd.read_csv(self.csv_path)

            # Convert Date/Datetime column to datetime
            date_col = None
            for col in ['Date', 'Datetime', 'date', 'datetime', 'timestamp', 'Timestamp']:
                if col in self.data.columns:
                    date_col = col
                    break

            if date_col:
                self.data[date_col] = pd.to_datetime(self.data[date_col])
                self.data.set_index(date_col, inplace=True)
            else:
                # If no date column, create a synthetic timestamp
                print("⚠️ No date column found, creating synthetic timestamps")
                start_date = datetime.now() - timedelta(days=len(self.data))
                self.data.index = pd.date_range(start=start_date, periods=len(self.data), freq='1min')

            # Ensure required columns exist (case-insensitive)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            col_mapping = {}
            for req_col in required_cols:
                for col in self.data.columns:
                    if col.lower() == req_col.lower():
                        col_mapping[col] = req_col
                        break

            if col_mapping:
                self.data.rename(columns=col_mapping, inplace=True)

            # Verify all required columns exist
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            if self.data.empty:
                raise ValueError(f"CSV file is empty: {self.csv_path}")

            print(f"✅ Successfully loaded {len(self.data)} data points from CSV")
            print(f"📅 Date range: {self.data.index[0]} to {self.data.index[-1]}")

        except Exception as e:
            print(f"❌ Error loading CSV for {self.symbol}: {e}")
            print(f"⚠️ Falling back to yfinance for {self.symbol}")
            # Fallback to yfinance if CSV fails
            self._fetch_from_yfinance()

    def _fetch_from_yfinance(self):
        """Fallback: fetch data from yfinance if CSV is not available."""
        try:
            print(f"📈 Fetching {self.symbol} from yfinance as fallback...")
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period="5d", interval="1m")

            if self.data.empty:
                raise ValueError(f"No data available for {self.symbol}")

            print(f"✅ Fetched {len(self.data)} data points from yfinance")

        except Exception as e:
            print(f"❌ yfinance fallback also failed: {e}")
            self._create_dummy_data()

    def _create_dummy_data(self):
        """Create dummy data as last resort."""
        print("🔧 Creating dummy data as last resort...")
        dates = pd.date_range(start=datetime.now() - timedelta(days=5), periods=1000, freq='1min')
        base_price = 150.0

        price_changes = np.random.normal(0, 0.1, len(dates))
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = max(prices[-1] + change, 1.0)
            prices.append(new_price)

        self.data = pd.DataFrame({
            'Open': prices,
            'High': [p + random.uniform(0, 0.5) for p in prices],
            'Low': [p - random.uniform(0, 0.5) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)

    def get_latest_price(self) -> float:
        """Get the most recent price."""
        if self.data is None or self.data.empty:
            return 150.0
        return float(self.data['Close'].iloc[-1])

    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Get historical data for the last N days."""
        if self.data is None or self.data.empty:
            return pd.DataFrame()

        cutoff_date = datetime.now() - timedelta(days=days)

        if hasattr(self.data.index, 'tz') and self.data.index.tz is not None:
            import pytz
            if self.data.index.tz:
                cutoff_date = cutoff_date.replace(tzinfo=pytz.UTC)
                cutoff_date = cutoff_date.astimezone(self.data.index.tz)
            else:
                cutoff_date = cutoff_date.replace(tzinfo=pytz.UTC)

        return self.data[self.data.index >= cutoff_date].copy()

    def stream(self, sleep_seconds: float = 1.0, replay_speed: float = 1.0) -> Iterator[TickData]:
        """
        Stream tick data from CSV.

        Args:
            sleep_seconds: Time to sleep between ticks
            replay_speed: Speed multiplier (1.0 = normal)

        Yields:
            TickData objects with market data
        """
        if self.data is None or self.data.empty:
            print("❌ No data available for streaming")
            return

        print(f"🚀 Starting CSV tick stream for {self.symbol}")
        print(f"⚡ Stream settings: sleep={sleep_seconds}s, speed={replay_speed}x")

        self.current_index = 0

        while self.current_index < len(self.data):
            row = self.data.iloc[self.current_index]
            timestamp = self.data.index[self.current_index]

            tick = TickData(
                timestamp=timestamp,
                open_price=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume']),
                symbol=self.symbol
            )

            yield tick

            self.current_index += 1

            if sleep_seconds > 0:
                time.sleep(sleep_seconds / replay_speed)

    def get_tick_at_index(self, index: int) -> Optional[TickData]:
        """Get a specific tick by index."""
        if self.data is None or self.data.empty or index >= len(self.data):
            return None

        row = self.data.iloc[index]
        timestamp = self.data.index[index]

        return TickData(
            timestamp=timestamp,
            open_price=float(row['Open']),
            high=float(row['High']),
            low=float(row['Low']),
            close=float(row['Close']),
            volume=int(row['Volume']),
            symbol=self.symbol
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded data."""
        if self.data is None or self.data.empty:
            return {"error": "No data available"}

        return {
            "symbol": self.symbol,
            "total_ticks": len(self.data),
            "date_range": {
                "start": str(self.data.index[0]),
                "end": str(self.data.index[-1])
            },
            "price_stats": {
                "min": float(self.data['Close'].min()),
                "max": float(self.data['Close'].max()),
                "mean": float(self.data['Close'].mean()),
                "std": float(self.data['Close'].std()),
                "current": float(self.data['Close'].iloc[-1])
            },
            "volume_stats": {
                "min": int(self.data['Volume'].min()),
                "max": int(self.data['Volume'].max()),
                "mean": float(self.data['Volume'].mean()),
                "total": int(self.data['Volume'].sum())
            }
        }


class YFinanceTickGenerator:
    """
    Generates market ticks from Yahoo Finance data.
    Supports both historical replay and live-like simulation.
    """
    
    def __init__(self, symbol: str, period: str = "30d", interval: str = "1m"):
        """
        Initialize the tick generator.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        """
        self.symbol = symbol.upper()
        self.period = period
        self.interval = interval
        self.data: Optional[pd.DataFrame] = None
        self.current_index = 0
        self._fetch_data()
        
    def _fetch_data(self):
        """Fetch historical data from Yahoo Finance."""
        try:
            print(f"📈 Fetching {self.symbol} data for period {self.period} with interval {self.interval}...")
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period, interval=self.interval)
            
            if self.data.empty:
                raise ValueError(f"No data available for {self.symbol}")
                
            print(f"✅ Successfully loaded {len(self.data)} data points for {self.symbol}")
            print(f"📅 Date range: {self.data.index[0]} to {self.data.index[-1]}")
            
        except Exception as e:
            print(f"❌ Error fetching data for {self.symbol}: {e}")
            # Create dummy data as fallback
            self._create_dummy_data()
            
    def _create_dummy_data(self):
        """Create dummy data for testing when Yahoo Finance is unavailable."""
        print("🔧 Creating dummy data for testing...")
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
        base_price = 150.0
        
        # Generate realistic price movements using random walk
        price_changes = np.random.normal(0, 0.1, len(dates))
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = max(prices[-1] + change, 1.0)  # Ensure price doesn't go negative
            prices.append(new_price)
            
        self.data = pd.DataFrame({
            'Open': prices,
            'High': [p + random.uniform(0, 0.5) for p in prices],
            'Low': [p - random.uniform(0, 0.5) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
    def get_latest_price(self) -> float:
        """Get the most recent price."""
        if self.data is None or self.data.empty:
            return 150.0  # Default price
        return float(self.data['Close'].iloc[-1])
        
    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Get historical data for the last N days."""
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        # Handle timezone-aware vs timezone-naive datetime comparison
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # If the data index has timezone info, make cutoff_date timezone-aware
        if hasattr(self.data.index, 'tz') and self.data.index.tz is not None:
            import pytz
            # Use the same timezone as the data, or UTC if data timezone is not available
            if self.data.index.tz:
                cutoff_date = cutoff_date.replace(tzinfo=pytz.UTC)
                cutoff_date = cutoff_date.astimezone(self.data.index.tz)
            else:
                cutoff_date = cutoff_date.replace(tzinfo=pytz.UTC)
        
        return self.data[self.data.index >= cutoff_date].copy()
        
    def stream(self, sleep_seconds: float = 1.0, replay_speed: float = 1.0) -> Iterator[TickData]:
        """
        Stream tick data in real-time simulation.
        
        Args:
            sleep_seconds: Time to sleep between ticks
            replay_speed: Speed multiplier for historical replay (1.0 = real-time)
            
        Yields:
            TickData objects with market data
        """
        if self.data is None or self.data.empty:
            print("❌ No data available for streaming")
            return
            
        print(f"🚀 Starting tick stream for {self.symbol}")
        print(f"⚡ Stream settings: sleep={sleep_seconds}s, speed={replay_speed}x")
        
        self.current_index = 0
        
        while self.current_index < len(self.data):
            row = self.data.iloc[self.current_index]
            timestamp = self.data.index[self.current_index]
            
            tick = TickData(
                timestamp=timestamp,
                open_price=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume']),
                symbol=self.symbol
            )
            
            yield tick
            
            self.current_index += 1
            
            # Sleep between ticks (adjusted by replay speed)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds / replay_speed)
                
    def get_tick_at_index(self, index: int) -> Optional[TickData]:
        """Get a specific tick by index."""
        if self.data is None or self.data.empty or index >= len(self.data):
            return None
            
        row = self.data.iloc[index]
        timestamp = self.data.index[index]
        
        return TickData(
            timestamp=timestamp,
            open_price=float(row['Open']),
            high=float(row['High']),
            low=float(row['Low']),
            close=float(row['Close']),
            volume=int(row['Volume']),
            symbol=self.symbol
        )
        
    def get_random_ticks(self, count: int = 100) -> List[TickData]:
        """Get random ticks for testing."""
        if self.data is None or self.data.empty:
            return []
            
        indices = np.random.choice(len(self.data), size=min(count, len(self.data)), replace=False)
        return [self.get_tick_at_index(idx) for idx in sorted(indices) if self.get_tick_at_index(idx)]
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded data."""
        if self.data is None or self.data.empty:
            return {"error": "No data available"}
            
        return {
            "symbol": self.symbol,
            "total_ticks": len(self.data),
            "date_range": {
                "start": str(self.data.index[0]),
                "end": str(self.data.index[-1])
            },
            "price_stats": {
                "min": float(self.data['Close'].min()),
                "max": float(self.data['Close'].max()),
                "mean": float(self.data['Close'].mean()),
                "std": float(self.data['Close'].std()),
                "current": float(self.data['Close'].iloc[-1])
            },
            "volume_stats": {
                "min": int(self.data['Volume'].min()),
                "max": int(self.data['Volume'].max()),
                "mean": float(self.data['Volume'].mean()),
                "total": int(self.data['Volume'].sum())
            }
        }


class LiveTickGenerator:
    """
    Generates live-like ticks with realistic price movements.
    Useful when Yahoo Finance is unavailable or for extended testing.
    """
    
    def __init__(self, symbol: str, initial_price: float = 150.0):
        self.symbol = symbol.upper()
        self.current_price = initial_price
        self.is_running = False
        self._lock = threading.Lock()
        
    def stream(self, sleep_seconds: float = 1.0, volatility: float = 0.1) -> Iterator[TickData]:
        """
        Generate live-like tick stream with realistic price movements.
        
        Args:
            sleep_seconds: Time between ticks
            volatility: Price volatility factor
        """
        print(f"🔴 Starting live tick generation for {self.symbol}")
        self.is_running = True
        
        while self.is_running:
            # Generate realistic price movement
            change_percent = np.random.normal(0, volatility / 100)
            price_change = self.current_price * change_percent
            
            new_price = max(self.current_price + price_change, 0.01)
            
            # Create realistic OHLC data
            high_offset = random.uniform(0, abs(price_change) + 0.05)
            low_offset = random.uniform(0, abs(price_change) + 0.05)
            
            tick = TickData(
                timestamp=datetime.now(),
                open_price=self.current_price,
                high=new_price + high_offset,
                low=new_price - low_offset,
                close=new_price,
                volume=random.randint(100, 5000),
                symbol=self.symbol
            )
            
            with self._lock:
                self.current_price = new_price
                
            yield tick
            
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
                
    def stop(self):
        """Stop the live tick generation."""
        self.is_running = False
        
    def get_current_price(self) -> float:
        """Get the current price."""
        with self._lock:
            return self.current_price


def display_stock_chart(symbol: str = "AAPL", days: int = 30):
    """
    Display a 30-day stock chart using matplotlib.
    This function will be called before starting the market simulation.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        print(f"📊 Displaying {days}-day chart for {symbol}...")
        
        # Fetch data
        tick_gen = YFinanceTickGenerator(symbol, period="1mo", interval="1d")
        data = tick_gen.get_historical_data(days=days)
        
        if data.empty:
            print("❌ No data available for chart")
            return
            
        # Create the chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart
        ax1.plot(data.index, data['Close'], linewidth=2, color='blue', label='Close Price')
        ax1.fill_between(data.index, data['Low'], data['High'], alpha=0.3, color='lightblue')
        ax1.set_title(f'{symbol} - {days} Day Price Chart', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Volume chart
        ax2.bar(data.index, data['Volume'], alpha=0.7, color='orange', label='Volume')
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Format dates
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days//10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save chart instead of showing it interactively
        chart_filename = f"{symbol.lower()}_30day_chart.png"
        plt.savefig(chart_filename, dpi=150, bbox_inches='tight')
        print(f"📊 Chart saved as: {chart_filename}")
        
        # Show chart briefly (non-blocking)
        plt.show(block=False)
        plt.pause(2)  # Display for 2 seconds
        plt.close()
        
        print("✅ Chart displayed. Continuing with simulation...")
        
    except ImportError:
        print("❌ matplotlib not available. Skipping chart display.")
    except Exception as e:
        print(f"❌ Error displaying chart: {e}")


if __name__ == "__main__":
    # Test the tick generator
    print("🧪 Testing YFinance Tick Generator")
    
    # Test with AAPL data
    tick_gen = YFinanceTickGenerator("AAPL", period="1d", interval="1m")
    
    # Display stats
    stats = tick_gen.get_stats()
    print(f"📊 Data Stats: {stats}")
    
    # Test streaming (first 10 ticks)
    print("\\n🔄 Testing tick stream:")
    for i, tick in enumerate(tick_gen.stream(sleep_seconds=0.1)):
        print(f"Tick {i+1}: {tick.symbol} @ ${tick.close:.2f} (Vol: {tick.volume:,})")
        if i >= 9:  # Stop after 10 ticks
            break
            
    # Test chart display
    display_stock_chart("AAPL", 30)
