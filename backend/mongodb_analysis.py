#!/usr/bin/env python3
"""
MongoDB Data Analysis Module
Provides comprehensive analysis of stock data for AI algorithm generation
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from database import get_db


# =============================================================================
# PATTERN EXTRACTION FUNCTIONS (For OpenRouter Prompt Enhancement)
# =============================================================================

def compute_mean_reversion(closes: np.ndarray, threshold: float = 0.005) -> float:
    """
    Compute mean reversion rate after significant price moves.
    
    Args:
        closes: Array of closing prices (minute-level data)
        threshold: Price move threshold (default 0.5% for minute data)
    
    Returns:
        Probability of price reverting after a threshold move
    """
    if len(closes) < 100:
        return 0.5
    
    # Calculate returns
    returns = np.diff(closes) / closes[:-1]
    
    # Find significant moves (> threshold)
    significant_up = np.where(returns > threshold)[0]
    significant_down = np.where(returns < -threshold)[0]
    
    reversions = 0
    total_events = 0
    
    # Check next 10 bars after significant up moves
    for idx in significant_up:
        if idx + 10 < len(returns):
            future_return = (closes[idx + 10] - closes[idx]) / closes[idx]
            if future_return < 0:  # Reverted
                reversions += 1
            total_events += 1
    
    # Check next 10 bars after significant down moves
    for idx in significant_down:
        if idx + 10 < len(returns):
            future_return = (closes[idx + 10] - closes[idx]) / closes[idx]
            if future_return > 0:  # Reverted
                reversions += 1
            total_events += 1
    
    return reversions / total_events if total_events > 0 else 0.5


def compute_momentum_persistence(closes: np.ndarray, lookback: int = 5) -> float:
    """
    Compute momentum persistence: probability that trend continues.
    
    Args:
        closes: Array of closing prices
        lookback: Number of bars to check for trend continuation
    
    Returns:
        Probability of momentum persisting
    """
    if len(closes) < lookback * 3:
        return 0.5
    
    continuations = 0
    total = 0
    
    for i in range(lookback, len(closes) - lookback):
        # Check if we had an up trend
        past_return = (closes[i] - closes[i - lookback]) / closes[i - lookback]
        future_return = (closes[i + lookback] - closes[i]) / closes[i]
        
        if abs(past_return) > 0.002:  # Only count significant moves
            if (past_return > 0 and future_return > 0) or (past_return < 0 and future_return < 0):
                continuations += 1
            total += 1
    
    return continuations / total if total > 0 else 0.5


def compute_volume_spike_returns(closes: np.ndarray, volumes: np.ndarray, spike_threshold: float = 2.0) -> Dict:
    """
    Compute average returns after volume spikes.
    
    Args:
        closes: Array of closing prices
        volumes: Array of volumes
        spike_threshold: Volume spike threshold (multiple of average)
    
    Returns:
        Dict with average returns after volume spikes
    """
    if len(closes) < 100 or len(volumes) < 100:
        return {'avg_return_1h': 0.0, 'avg_return_4h': 0.0, 'spike_count': 0}
    
    # Calculate rolling average volume (using 60 bars = 1 hour)
    window = 60
    avg_volumes = np.convolve(volumes, np.ones(window)/window, mode='valid')
    
    # Find volume spikes
    spikes = []
    for i in range(window, len(volumes) - 240):  # Leave room for 4h returns
        current_vol = volumes[i]
        recent_avg = avg_volumes[i - window] if i - window < len(avg_volumes) else np.mean(volumes[max(0, i-window):i])
        if recent_avg > 0 and current_vol > recent_avg * spike_threshold:
            spikes.append(i)
    
    if not spikes:
        return {'avg_return_1h': 0.0, 'avg_return_4h': 0.0, 'spike_count': 0}
    
    returns_1h = []
    returns_4h = []
    
    for idx in spikes:
        if idx + 60 < len(closes):
            ret_1h = (closes[idx + 60] - closes[idx]) / closes[idx]
            returns_1h.append(ret_1h)
        if idx + 240 < len(closes):
            ret_4h = (closes[idx + 240] - closes[idx]) / closes[idx]
            returns_4h.append(ret_4h)
    
    return {
        'avg_return_1h': float(np.mean(returns_1h)) if returns_1h else 0.0,
        'avg_return_4h': float(np.mean(returns_4h)) if returns_4h else 0.0,
        'spike_count': len(spikes)
    }


def compute_hourly_volatility(datetimes: List[str], closes: np.ndarray) -> Dict[int, float]:
    """
    Compute volatility by hour of day.
    
    Args:
        datetimes: List of datetime strings
        closes: Array of closing prices
    
    Returns:
        Dict mapping hour -> standard deviation of returns
    """
    hourly_returns = defaultdict(list)
    
    for i in range(1, len(closes)):
        try:
            # Parse hour from datetime string "YYYY-MM-DD HH:MM:SS"
            dt_str = datetimes[i]
            hour = int(dt_str.split()[1].split(':')[0])
            
            # Calculate return
            ret = (closes[i] - closes[i-1]) / closes[i-1]
            hourly_returns[hour].append(ret)
        except (IndexError, ValueError):
            continue
    
    # Compute std dev for each hour
    hourly_volatility = {}
    for hour, returns in hourly_returns.items():
        if len(returns) > 10:
            hourly_volatility[hour] = float(np.std(returns))
    
    return hourly_volatility


def compute_gap_statistics(datetimes: List[str], opens: np.ndarray, closes: np.ndarray) -> Dict:
    """
    Compute overnight gap statistics.
    
    Args:
        datetimes: List of datetime strings
        opens: Array of opening prices
        closes: Array of closing prices
    
    Returns:
        Dict with gap statistics
    """
    gaps = []
    gap_fills = 0
    
    prev_close = None
    prev_date = None
    
    for i, dt_str in enumerate(datetimes):
        try:
            date = dt_str.split()[0]
            
            if prev_date and date != prev_date:
                # New day - calculate gap
                if prev_close:
                    gap = (opens[i] - prev_close) / prev_close
                    gaps.append(gap)
                    
                    # Check if gap filled within first hour (60 minutes)
                    day_start = i
                    for j in range(i, min(i + 60, len(closes))):
                        if datetimes[j].split()[0] != date:
                            break
                        # Gap up filled if price goes below previous close
                        # Gap down filled if price goes above previous close
                        if gap > 0 and closes[j] <= prev_close:
                            gap_fills += 1
                            break
                        elif gap < 0 and closes[j] >= prev_close:
                            gap_fills += 1
                            break
            
            prev_date = date
            prev_close = closes[i]
        except (IndexError, ValueError):
            continue
    
    if not gaps:
        return {
            'avg_gap': 0.0,
            'avg_gap_up': 0.0,
            'avg_gap_down': 0.0,
            'gap_fill_rate': 0.0,
            'total_gaps': 0
        }
    
    gaps = np.array(gaps)
    gap_ups = gaps[gaps > 0]
    gap_downs = gaps[gaps < 0]
    
    return {
        'avg_gap': float(np.mean(np.abs(gaps))),
        'avg_gap_up': float(np.mean(gap_ups)) if len(gap_ups) > 0 else 0.0,
        'avg_gap_down': float(np.mean(gap_downs)) if len(gap_downs) > 0 else 0.0,
        'gap_fill_rate': gap_fills / len(gaps) if gaps.size > 0 else 0.0,
        'total_gaps': len(gaps)
    }


def compute_day_of_week_patterns(datetimes: List[str], closes: np.ndarray) -> Dict[str, float]:
    """
    Compute average returns by day of week.
    
    Args:
        datetimes: List of datetime strings
        closes: Array of closing prices
    
    Returns:
        Dict mapping day name -> average return
    """
    # Group returns by day of week
    day_returns = defaultdict(list)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    
    prev_date = None
    prev_close = None
    
    for i, dt_str in enumerate(datetimes):
        try:
            date_str = dt_str.split()[0]
            
            if prev_date and date_str != prev_date:
                # Calculate daily return
                if prev_close:
                    daily_ret = (closes[i] - prev_close) / prev_close
                    # Parse date to get day of week
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    day_idx = dt.weekday()
                    if day_idx < 5:  # Only weekdays
                        day_returns[days[day_idx]].append(daily_ret)
            
            prev_date = date_str
            prev_close = closes[i]
        except (IndexError, ValueError):
            continue
    
    # Compute averages
    day_avg_returns = {}
    for day, returns in day_returns.items():
        if returns:
            day_avg_returns[day] = float(np.mean(returns))
    
    return day_avg_returns


def extract_real_patterns(ticker: str, end_date: str = "2025-11-24") -> Optional[Dict]:
    """
    Extract real statistical patterns from historical minute-level data.
    Run ONCE and store/cache results for prompt enhancement.
    
    Args:
        ticker: Stock ticker symbol
        end_date: End date for analysis (simulation start date)
    
    Returns:
        Dict containing computed patterns or None if no data
    """
    print(f"\n🔍 Extracting real patterns for {ticker}...")
    
    try:
        db = get_db()
        collection = db[f"{ticker}_historical"]
        
        # Fetch ALL data up to end_date
        cursor = collection.find({}).sort('datetime', 1)
        data = list(cursor)
        
        if not data:
            print(f"⚠️ No data found for {ticker}")
            return None
        
        # Filter by end_date
        if end_date:
            end_datetime = f"{end_date} 23:59:59"
            data = [record for record in data if record['datetime'] <= end_datetime]
        
        if not data:
            print(f"⚠️ No data found for {ticker} (all filtered out)")
            return None
        
        print(f"   Analyzing {len(data)} minute records...")
        
        # Extract arrays
        datetimes = [d['datetime'] for d in data]
        closes = np.array([d['close'] for d in data])
        opens = np.array([d['open'] for d in data])
        volumes = np.array([d['volume'] for d in data])
        
        # Compute all patterns
        patterns = {
            'ticker': ticker,
            'total_minutes': len(data),
            'mean_reversion_rate': compute_mean_reversion(closes),
            'momentum_persistence': compute_momentum_persistence(closes),
            'volatility_by_hour': compute_hourly_volatility(datetimes, closes),
            'volume_spike_returns': compute_volume_spike_returns(closes, volumes),
            'gap_statistics': compute_gap_statistics(datetimes, opens, closes),
            'day_of_week_returns': compute_day_of_week_patterns(datetimes, closes),
        }
        
        print(f"   ✅ Pattern extraction complete")
        print(f"   - Mean reversion rate: {patterns['mean_reversion_rate']:.1%}")
        print(f"   - Momentum persistence: {patterns['momentum_persistence']:.1%}")
        
        return patterns
        
    except Exception as e:
        print(f"   ❌ Error extracting patterns: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_mongodb_data(ticker: str, end_date: str = "2025-11-24") -> Optional[Dict]:
    """
    Comprehensive analysis of ALL MongoDB data for a ticker

    Args:
        ticker: Stock ticker symbol
        end_date: End date for analysis (simulation start date)

    Returns:
        Dict containing comprehensive analysis or None if no data
    """
    print(f"\n📊 Analyzing MongoDB data for {ticker}...")

    try:
        db = get_db()
        collection = db[f"{ticker}_historical"]
        print(f"   Connected to database: {db.name}")
        print(f"   Collection: {collection.name}")

        # Check collection exists and has data
        count = collection.count_documents({})
        print(f"   Total documents in collection: {count}")

        # Fetch ALL data
        # Note: datetime format is "YYYY-MM-DD HH:MM:SS" (string)
        # We filter by date in Python since string comparison works: "2025-01-02 09:30:00" <= "2025-11-24 23:59:59"
        cursor = collection.find({}).sort('datetime', 1)
        data = list(cursor)
        print(f"   Fetched {len(data)} records from database")
    except Exception as e:
        print(f"   ❌ Error connecting to database: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Filter by end_date if specified
    if end_date and data:
        print(f"   Filtering by end_date: {end_date}")
        # Add time component to end_date for proper comparison
        end_datetime = f"{end_date} 23:59:59"
        print(f"   End datetime: {end_datetime}")

        # Debug: Show first and last dates
        if data:
            print(f"   First record date: {data[0]['datetime']}")
            print(f"   Last record date: {data[-1]['datetime']}")

        original_count = len(data)
        data = [record for record in data if record['datetime'] <= end_datetime]
        print(f"   After filtering: {len(data)} records (removed {original_count - len(data)})")

    if not data:
        print(f"⚠️ No data found for {ticker} (all records filtered out)")
        return None

    print(f"   Found {len(data)} records")

    # Extract arrays for calculations
    datetimes = [d['datetime'] for d in data]
    closes = np.array([d['close'] for d in data])
    highs = np.array([d['high'] for d in data])
    lows = np.array([d['low'] for d in data])
    opens = np.array([d['open'] for d in data])
    volumes = np.array([d['volume'] for d in data])

    # Calculate all analysis components
    analysis = {
        'ticker': ticker,
        'date_range': {
            'start': datetimes[0],
            'end': datetimes[-1]
        },
        'total_days': len(set([d.split()[0] for d in datetimes])),
        'total_minutes': len(data),
        'price_stats': calculate_price_stats(closes, highs, lows),
        'trends': calculate_trends(closes),
        'volatility': calculate_volatility(closes, highs, lows),
        'momentum': calculate_momentum(closes),
        'volume': analyze_volume(volumes),
        'levels': find_support_resistance(closes, highs, lows),
        'regime': detect_market_regime(closes, highs, lows),
        'patterns': analyze_patterns(closes),
    }

    print(f"   ✅ Analysis complete")
    return analysis


def calculate_price_stats(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict:
    """Calculate comprehensive price statistics"""
    return {
        'min': float(np.min(closes)),
        'max': float(np.max(closes)),
        'mean': float(np.mean(closes)),
        'median': float(np.median(closes)),
        'std': float(np.std(closes)),
        'current': float(closes[-1]),
        'range': float(np.max(closes) - np.min(closes)),
        'high_90d': float(np.max(closes[-390*90:])) if len(closes) >= 390*90 else float(np.max(closes)),
        'low_90d': float(np.min(closes[-390*90:])) if len(closes) >= 390*90 else float(np.min(closes)),
        'high_30d': float(np.max(closes[-390*30:])) if len(closes) >= 390*30 else float(np.max(closes)),
        'low_30d': float(np.min(closes[-390*30:])) if len(closes) >= 390*30 else float(np.min(closes)),
    }


def calculate_trends(closes: np.ndarray) -> Dict:
    """Calculate trend indicators (SMA, EMA, trend direction)"""
    # Calculate SMAs
    sma_20 = float(np.mean(closes[-20*390:])) if len(closes) >= 20*390 else float(np.mean(closes))
    sma_50 = float(np.mean(closes[-50*390:])) if len(closes) >= 50*390 else float(np.mean(closes))
    sma_200 = float(np.mean(closes[-200*390:])) if len(closes) >= 200*390 else float(np.mean(closes))

    # Determine trend direction
    if sma_20 > sma_50:
        direction = "BULLISH"
        strength = min(10, int(((sma_20 - sma_50) / sma_50) * 100))
    elif sma_20 < sma_50:
        direction = "BEARISH"
        strength = min(10, int(((sma_50 - sma_20) / sma_50) * 100))
    else:
        direction = "NEUTRAL"
        strength = 0

    # Calculate momentum (rate of change)
    if len(closes) >= 390*10:
        momentum = float((closes[-1] - closes[-390*10]) / closes[-390*10])
    else:
        momentum = 0.0

    return {
        'sma_20': sma_20,
        'sma_50': sma_50,
        'sma_200': sma_200,
        'direction': direction,
        'strength': max(1, strength),
        'momentum': momentum
    }


def calculate_volatility(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict:
    """Calculate volatility metrics (ATR, Bollinger Bands, std dev)"""
    # Standard deviation
    std_dev = float(np.std(closes[-390*20:])) if len(closes) >= 390*20 else float(np.std(closes))
    mean_price = float(np.mean(closes[-390*20:])) if len(closes) >= 390*20 else float(np.mean(closes))
    std_dev_pct = std_dev / mean_price if mean_price > 0 else 0

    # ATR calculation (simplified)
    if len(closes) >= 390*14:
        true_ranges = []
        for i in range(-390*14, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]) if i > -len(closes) else 0,
                abs(lows[i] - closes[i-1]) if i > -len(closes) else 0
            )
            true_ranges.append(tr)
        atr_14 = float(np.mean(true_ranges))
        atr_20 = float(np.mean(true_ranges[-390*20:])) if len(true_ranges) >= 390*20 else atr_14
    else:
        atr_14 = float(np.mean(highs - lows))
        atr_20 = atr_14

    # Determine volatility regime
    if std_dev_pct < 0.015:
        regime = "LOW"
    elif std_dev_pct < 0.025:
        regime = "NORMAL"
    else:
        regime = "HIGH"

    # Bollinger Band width
    sma_20 = np.mean(closes[-390*20:]) if len(closes) >= 390*20 else np.mean(closes)
    std_20 = np.std(closes[-390*20:]) if len(closes) >= 390*20 else np.std(closes)
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    bb_width = float((bb_upper - bb_lower) / sma_20) if sma_20 > 0 else 0

    return {
        'atr_14': atr_14,
        'atr_20': atr_20,
        'std_dev': std_dev_pct,
        'regime': regime,
        'bb_width': bb_width,
        'bb_upper': float(bb_upper),
        'bb_lower': float(bb_lower)
    }


def calculate_momentum(closes: np.ndarray) -> Dict:
    """Calculate momentum indicators (RSI, MACD, ROC)"""
    # RSI calculation
    if len(closes) >= 390*14:
        deltas = np.diff(closes[-390*15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-390*14:])
        avg_loss = np.mean(losses[-390*14:])
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi_14 = 100 - (100 / (1 + rs)) if rs > 0 else 50
    else:
        rsi_14 = 50.0

    # MACD calculation (12, 26, 9)
    if len(closes) >= 390*26:
        ema_12 = calculate_ema(closes, 12*390)
        ema_26 = calculate_ema(closes, 26*390)
        macd = ema_12 - ema_26

        # MACD signal line (9-period EMA of MACD)
        macd_signal = ema_12 - ema_26  # Simplified
        macd_hist = macd - macd_signal
    else:
        macd = 0.0
        macd_signal = 0.0
        macd_hist = 0.0

    # Rate of Change
    if len(closes) >= 390*10:
        roc_10 = float((closes[-1] - closes[-390*10]) / closes[-390*10])
    else:
        roc_10 = 0.0

    if len(closes) >= 390*20:
        roc_20 = float((closes[-1] - closes[-390*20]) / closes[-390*20])
    else:
        roc_20 = 0.0

    return {
        'rsi_14': float(rsi_14),
        'macd': float(macd),
        'macd_signal': float(macd_signal),
        'macd_hist': float(macd_hist),
        'roc_10': roc_10,
        'roc_20': roc_20
    }


def calculate_ema(data: np.ndarray, period: int) -> float:
    """Calculate Exponential Moving Average"""
    if len(data) < period:
        return float(np.mean(data))

    multiplier = 2 / (period + 1)
    ema = np.mean(data[-period:])  # Start with SMA

    for price in data[-period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))

    return float(ema)


def analyze_volume(volumes: np.ndarray) -> Dict:
    """Analyze volume patterns and trends"""
    # Calculate daily volumes (aggregate minute data)
    daily_volumes = []
    current_sum = 0
    for i, vol in enumerate(volumes):
        current_sum += vol
        if (i + 1) % 390 == 0:  # End of trading day
            daily_volumes.append(current_sum)
            current_sum = 0

    if not daily_volumes:
        daily_volumes = [float(np.sum(volumes))]

    avg_daily = float(np.mean(daily_volumes))
    recent_20d = float(np.mean(daily_volumes[-20:])) if len(daily_volumes) >= 20 else avg_daily

    # Determine trend
    if recent_20d > avg_daily * 1.1:
        trend = "INCREASING"
    elif recent_20d < avg_daily * 0.9:
        trend = "DECREASING"
    else:
        trend = "STABLE"

    # Count high volume days
    high_volume_threshold = avg_daily * 1.5
    high_volume_days = sum(1 for v in daily_volumes[-90:] if v > high_volume_threshold)

    return {
        'avg_daily': avg_daily,
        'recent_20d': recent_20d,
        'trend': trend,
        'high_volume_days': high_volume_days,
        'total_volume': float(np.sum(volumes))
    }


def find_support_resistance(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict:
    """Find key support and resistance levels"""
    # Use last 90 days of data
    lookback = min(390*90, len(closes))
    recent_closes = closes[-lookback:]
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]

    # Find local minima for support (simplified approach)
    support_levels = []
    for i in range(20, len(recent_lows)-20):
        if recent_lows[i] == np.min(recent_lows[i-20:i+20]):
            support_levels.append(float(recent_lows[i]))

    # Find local maxima for resistance
    resistance_levels = []
    for i in range(20, len(recent_highs)-20):
        if recent_highs[i] == np.max(recent_highs[i-20:i+20]):
            resistance_levels.append(float(recent_highs[i]))

    # Cluster levels (group similar levels together)
    def cluster_levels(levels, tolerance=0.02):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            if level - current_cluster[-1] < current_cluster[-1] * tolerance:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        clusters.append(np.mean(current_cluster))
        return clusters

    support_clusters = cluster_levels(support_levels)
    resistance_clusters = cluster_levels(resistance_levels)

    # Get top 3 levels
    support_strong = float(support_clusters[0]) if support_clusters else float(np.min(recent_lows))
    support_2 = float(support_clusters[1]) if len(support_clusters) > 1 else support_strong * 0.98
    support_3 = float(support_clusters[2]) if len(support_clusters) > 2 else support_strong * 0.96

    resistance_strong = float(resistance_clusters[-1]) if resistance_clusters else float(np.max(recent_highs))
    resistance_2 = float(resistance_clusters[-2]) if len(resistance_clusters) > 1 else resistance_strong * 1.02
    resistance_3 = float(resistance_clusters[-3]) if len(resistance_clusters) > 2 else resistance_strong * 1.04

    return {
        'support_strong': support_strong,
        'support_strong_tests': len([s for s in support_levels if abs(s - support_strong) < support_strong * 0.01]),
        'support_2': support_2,
        'support_3': support_3,
        'resistance_strong': resistance_strong,
        'resistance_strong_tests': len([r for r in resistance_levels if abs(r - resistance_strong) < resistance_strong * 0.01]),
        'resistance_2': resistance_2,
        'resistance_3': resistance_3
    }


def detect_market_regime(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict:
    """Detect current market regime (trending, choppy, breakout, etc.)"""
    lookback = min(390*60, len(closes))
    recent_closes = closes[-lookback:]
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]

    # Calculate trend strength
    sma_20 = np.mean(recent_closes[-390*20:]) if len(recent_closes) >= 390*20 else np.mean(recent_closes)
    sma_50 = np.mean(recent_closes[-390*50:]) if len(recent_closes) >= 390*50 else np.mean(recent_closes)

    # Choppiness Index (simplified)
    atr = np.mean(recent_highs - recent_lows)
    high_low_range = np.max(recent_highs) - np.min(recent_lows)
    choppiness = (atr / high_low_range * 100) if high_low_range > 0 else 50

    # Directional Movement
    directional_movement = abs(sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0

    # ATR as percentage of price
    atr_percentage = atr / recent_closes[-1] if recent_closes[-1] > 0 else 0

    # Determine regime
    if choppiness > 62:
        current_regime = "CHOPPY"
    elif choppiness < 38 and sma_20 > sma_50:
        current_regime = "TRENDING_UP"
    elif choppiness < 38 and sma_20 < sma_50:
        current_regime = "TRENDING_DOWN"
    elif atr_percentage > 0.03:
        current_regime = "BREAKOUT"
    else:
        current_regime = "CONSOLIDATION"

    return {
        'current': current_regime,
        'choppiness': float(choppiness),
        'directional_movement': float(directional_movement),
        'atr_percentage': float(atr_percentage)
    }


def analyze_patterns(closes: np.ndarray) -> Dict:
    """Analyze historical price patterns"""
    # Calculate daily returns
    daily_closes = []
    for i in range(0, len(closes), 390):
        if i < len(closes):
            daily_closes.append(closes[i])

    daily_closes = np.array(daily_closes)
    if len(daily_closes) < 2:
        return {
            'win_rate': 0.5,
            'avg_up_day': 0.0,
            'avg_down_day': 0.0,
            'max_gain': 0.0,
            'max_loss': 0.0,
            'max_consecutive_up': 0,
            'max_consecutive_down': 0
        }

    daily_returns = np.diff(daily_closes) / daily_closes[:-1]

    # Win rate
    up_days = np.sum(daily_returns > 0)
    total_days = len(daily_returns)
    win_rate = up_days / total_days if total_days > 0 else 0.5

    # Average up/down days
    avg_up_day = float(np.mean(daily_returns[daily_returns > 0])) if np.any(daily_returns > 0) else 0.0
    avg_down_day = float(np.mean(daily_returns[daily_returns < 0])) if np.any(daily_returns < 0) else 0.0

    # Max gain/loss
    max_gain = float(np.max(daily_returns)) if len(daily_returns) > 0 else 0.0
    max_loss = float(np.min(daily_returns)) if len(daily_returns) > 0 else 0.0

    # Consecutive days
    max_consecutive_up = 0
    max_consecutive_down = 0
    current_up = 0
    current_down = 0

    for ret in daily_returns:
        if ret > 0:
            current_up += 1
            current_down = 0
            max_consecutive_up = max(max_consecutive_up, current_up)
        elif ret < 0:
            current_down += 1
            current_up = 0
            max_consecutive_down = max(max_consecutive_down, current_down)

    return {
        'win_rate': float(win_rate),
        'avg_up_day': avg_up_day,
        'avg_down_day': avg_down_day,
        'max_gain': max_gain,
        'max_loss': max_loss,
        'max_consecutive_up': max_consecutive_up,
        'max_consecutive_down': max_consecutive_down
    }


def generate_compressed_historical(ticker: str, end_date: str = "2025-11-24") -> Optional[Dict]:
    """
    Generate compressed daily historical data for embedding in algorithms
    Reduces 87K minute records to ~300 daily records

    Args:
        ticker: Stock ticker symbol
        end_date: End date for data (simulation start date)

    Returns:
        Dict with compressed daily OHLCV data
    """
    print(f"\n📦 Generating compressed historical data for {ticker}...")

    try:
        db = get_db()
        collection = db[f"{ticker}_historical"]
        print(f"   Connected to database: {db.name}")
        print(f"   Collection: {collection.name}")

        # Check collection exists and has data
        count = collection.count_documents({})
        print(f"   Total documents in collection: {count}")

        # Fetch all data
        cursor = collection.find({}).sort('datetime', 1)
        data = list(cursor)
        print(f"   Fetched {len(data)} records from database")
    except Exception as e:
        print(f"   ❌ Error connecting to database: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Filter by end_date if specified
    if end_date and data:
        print(f"   Filtering by end_date: {end_date}")
        end_datetime = f"{end_date} 23:59:59"
        print(f"   End datetime: {end_datetime}")

        # Debug: Show first and last dates
        if data:
            print(f"   First record date: {data[0]['datetime']}")
            print(f"   Last record date: {data[-1]['datetime']}")

        original_count = len(data)
        data = [record for record in data if record['datetime'] <= end_datetime]
        print(f"   After filtering: {len(data)} records (removed {original_count - len(data)})")

    if not data:
        print(f"⚠️ No data found for {ticker} (all records filtered out)")
        return None

    # Aggregate to daily data
    daily_data = {}
    for record in data:
        date = record['datetime'].split()[0]  # Get just date part (YYYY-MM-DD)

        if date not in daily_data:
            daily_data[date] = {
                'open': record['open'],
                'high': record['high'],
                'low': record['low'],
                'close': record['close'],
                'volume': record['volume']
            }
        else:
            # Update daily aggregates
            daily_data[date]['high'] = max(daily_data[date]['high'], record['high'])
            daily_data[date]['low'] = min(daily_data[date]['low'], record['low'])
            daily_data[date]['close'] = record['close']  # Last close of day
            daily_data[date]['volume'] += record['volume']

    # Convert to lists
    dates = sorted(daily_data.keys())
    closes = [daily_data[d]['close'] for d in dates]
    opens = [daily_data[d]['open'] for d in dates]
    highs = [daily_data[d]['high'] for d in dates]
    lows = [daily_data[d]['low'] for d in dates]
    volumes = [daily_data[d]['volume'] for d in dates]

    print(f"   Compressed {len(data)} minute records → {len(dates)} daily records")
    print(f"   ✅ Compression complete")

    return {
        'dates': dates,
        'closes': closes,
        'opens': opens,
        'highs': highs,
        'lows': lows,
        'volumes': volumes
    }


def get_analysis_data(ticker: str = "AAPL") -> Optional[Dict]:
    """
    Get comprehensive analysis data for a ticker including:
    - Price patterns
    - Volatility analysis
    - Trend detection
    - Support/Resistance levels
    - Best trading hours
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dict containing comprehensive analysis data or None if no data
    """
    try:
        db = get_db()
        historical_collection = db[f"{ticker}_historical"]
        
        # Get all historical data
        cursor = historical_collection.find({}).sort('datetime', 1)
        records = list(cursor)
        
        if not records:
            print(f"⚠️ No historical data found for {ticker}")
            return None
        
        print(f"📊 Analyzing {len(records)} records for {ticker}")
        
        # Extract prices
        prices = [r['close'] for r in records]
        highs = [r['high'] for r in records]
        lows = [r['low'] for r in records]
        volumes = [r.get('volume', 0) for r in records]
        datetimes = [r['datetime'] for r in records]
        
        # Calculate patterns
        patterns = {}
        
        # 1. Overall trend
        if len(prices) > 100:
            start_price = sum(prices[:20]) / 20  # First 20 avg
            end_price = sum(prices[-20:]) / 20   # Last 20 avg
            trend_pct = (end_price - start_price) / start_price * 100
            patterns['overall_trend'] = 'BULLISH' if trend_pct > 1 else 'BEARISH' if trend_pct < -1 else 'SIDEWAYS'
            patterns['trend_strength'] = abs(trend_pct)
        
        # 2. Volatility analysis
        if len(prices) > 50:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] * 100 for i in range(1, len(prices))]
            patterns['avg_volatility'] = sum(abs(r) for r in returns) / len(returns)
            patterns['max_move_up'] = max(returns)
            patterns['max_move_down'] = min(returns)
            
            # Categorize volatility
            if patterns['avg_volatility'] > 0.5:
                patterns['volatility_category'] = 'HIGH'
            elif patterns['avg_volatility'] > 0.2:
                patterns['volatility_category'] = 'MEDIUM'
            else:
                patterns['volatility_category'] = 'LOW'
        
        # 3. Mean reversion analysis
        if len(prices) > 100:
            reversion_count = 0
            continuation_count = 0
            
            for i in range(10, len(prices) - 5):
                move = (prices[i] - prices[i-10]) / prices[i-10] * 100
                next_move = (prices[i+5] - prices[i]) / prices[i] * 100
                
                if abs(move) > 0.3:  # Significant move
                    if (move > 0 and next_move < 0) or (move < 0 and next_move > 0):
                        reversion_count += 1
                    else:
                        continuation_count += 1
            
            total = reversion_count + continuation_count
            if total > 0:
                patterns['mean_reversion_rate'] = reversion_count / total * 100
                patterns['momentum_rate'] = continuation_count / total * 100
                patterns['strategy_hint'] = 'MEAN_REVERSION' if patterns['mean_reversion_rate'] > 55 else 'MOMENTUM'
        
        # 4. Best trading hours (if minute data)
        hour_performance = {}
        for i, dt_str in enumerate(datetimes):
            try:
                if ' ' in dt_str:
                    hour = int(dt_str.split(' ')[1].split(':')[0])
                    if i > 0:
                        ret = (prices[i] - prices[i-1]) / prices[i-1] * 100
                        if hour not in hour_performance:
                            hour_performance[hour] = []
                        hour_performance[hour].append(abs(ret))
            except:
                pass
        
        if hour_performance:
            avg_by_hour = {h: sum(v)/len(v) for h, v in hour_performance.items()}
            patterns['most_volatile_hour'] = max(avg_by_hour, key=avg_by_hour.get)
            patterns['least_volatile_hour'] = min(avg_by_hour, key=avg_by_hour.get)
            patterns['hourly_volatility'] = avg_by_hour
        
        # 5. Support and Resistance levels
        if len(prices) > 50:
            sorted_prices = sorted(prices)
            patterns['support_level'] = sorted_prices[int(len(sorted_prices) * 0.1)]  # 10th percentile
            patterns['resistance_level'] = sorted_prices[int(len(sorted_prices) * 0.9)]  # 90th percentile
            patterns['current_price'] = prices[-1]
            patterns['price_position'] = 'NEAR_SUPPORT' if prices[-1] < patterns['support_level'] * 1.02 else \
                                        'NEAR_RESISTANCE' if prices[-1] > patterns['resistance_level'] * 0.98 else 'MIDDLE'
        
        # 6. Day-of-week analysis
        dow_returns = {0: [], 1: [], 2: [], 3: [], 4: []}  # Mon-Fri
        for i, dt_str in enumerate(datetimes):
            try:
                dt = datetime.strptime(dt_str.split(' ')[0], '%Y-%m-%d')
                dow = dt.weekday()
                if dow < 5 and i > 0:
                    ret = (prices[i] - prices[i-1]) / prices[i-1] * 100
                    dow_returns[dow].append(ret)
            except:
                pass
        
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        patterns['day_performance'] = {}
        for dow, returns in dow_returns.items():
            if returns:
                patterns['day_performance'][dow_names[dow]] = {
                    'avg_return': sum(returns) / len(returns),
                    'win_rate': len([r for r in returns if r > 0]) / len(returns) * 100
                }
        
        return {
            'ticker': ticker,
            'record_count': len(records),
            'date_range': f"{datetimes[0]} to {datetimes[-1]}",
            'prices': prices,
            'patterns': patterns
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def format_patterns_for_prompt(patterns: Dict) -> str:
    """
    Format patterns dict into a readable string for the AI prompt.
    
    Args:
        patterns: Dict of computed patterns from get_analysis_data
    
    Returns:
        Formatted string for embedding in AI prompts
    """
    if not patterns:
        return "No pattern data available."
    
    lines = []
    
    # Overall trend
    if 'overall_trend' in patterns:
        lines.append(f"📈 OVERALL TREND: {patterns['overall_trend']} (strength: {patterns.get('trend_strength', 0):.1f}%)")
    
    # Volatility
    if 'volatility_category' in patterns:
        lines.append(f"📊 VOLATILITY: {patterns['volatility_category']} (avg: {patterns.get('avg_volatility', 0):.2f}%)")
        lines.append(f"   Max up: +{patterns.get('max_move_up', 0):.2f}% | Max down: {patterns.get('max_move_down', 0):.2f}%")
    
    # Strategy hint
    if 'strategy_hint' in patterns:
        lines.append(f"💡 SUGGESTED STRATEGY: {patterns['strategy_hint']}")
        if patterns['strategy_hint'] == 'MEAN_REVERSION':
            lines.append(f"   → Price tends to reverse after big moves ({patterns.get('mean_reversion_rate', 0):.0f}% of the time)")
        else:
            lines.append(f"   → Price tends to continue in direction ({patterns.get('momentum_rate', 0):.0f}% of the time)")
    
    # Support/Resistance
    if 'support_level' in patterns:
        lines.append(f"🎯 KEY LEVELS:")
        lines.append(f"   Support: ${patterns['support_level']:.2f}")
        lines.append(f"   Resistance: ${patterns['resistance_level']:.2f}")
        lines.append(f"   Current: ${patterns['current_price']:.2f} ({patterns['price_position']})")
    
    # Best hours
    if 'most_volatile_hour' in patterns:
        lines.append(f"⏰ BEST TRADING HOURS:")
        lines.append(f"   Most active: {patterns['most_volatile_hour']}:00")
        lines.append(f"   Least active: {patterns['least_volatile_hour']}:00")
    
    # Day of week
    if 'day_performance' in patterns and patterns['day_performance']:
        lines.append(f"📅 DAY PERFORMANCE:")
        for day, perf in patterns['day_performance'].items():
            lines.append(f"   {day}: avg {perf['avg_return']:+.3f}%, win rate {perf['win_rate']:.0f}%")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the analysis
    test_ticker = "AAPL"
    
    # Test new get_analysis_data function
    print("\n" + "="*60)
    print("Testing get_analysis_data()")
    print("="*60)
    analysis_data = get_analysis_data(test_ticker)
    if analysis_data:
        print(f"\nDate Range: {analysis_data['date_range']}")
        print(f"Records: {analysis_data['record_count']}")
        print("\nFormatted Patterns for AI Prompt:")
        print("-"*40)
        print(format_patterns_for_prompt(analysis_data['patterns']))
    
    # Test original analyze_mongodb_data function
    print("\n" + "="*60)
    print("Testing analyze_mongodb_data()")
    print("="*60)
    analysis = analyze_mongodb_data(test_ticker)

    if analysis:
        print(f"Date Range: {analysis['date_range']['start']} to {analysis['date_range']['end']}")
        print(f"Total Days: {analysis['total_days']}")
        print(f"Total Minutes: {analysis['total_minutes']}")
        print(f"Current Price: ${analysis['price_stats']['current']:.2f}")
        print(f"Trend: {analysis['trends']['direction']}")
        print(f"RSI: {analysis['momentum']['rsi_14']:.1f}")
        print(f"Market Regime: {analysis['regime']['current']}")

    compressed = generate_compressed_historical(test_ticker)
    if compressed:
        print(f"\nCompressed to {len(compressed['closes'])} daily records")
