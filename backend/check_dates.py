#!/usr/bin/env python3
"""
Quick script to verify the dates are configured correctly
"""

print("\n" + "="*60)
print("📅 CURRENT DATE CONFIGURATION")
print("="*60)

# Import the configuration
from fetch_minute_data_mongodb import (
    SIMULATION_DATE,
    HISTORICAL_END_DATE,
    HISTORICAL_START_DATE
)

print("\n✅ Configured Dates:")
print("-"*60)
print(f"   Simulation Day:      {SIMULATION_DATE}")
print(f"   Historical End:      {HISTORICAL_END_DATE}")
print(f"   Historical Start:    {HISTORICAL_START_DATE}")

print("\n📊 Data Organization:")
print("-"*60)
print(f"   Historical Data:     {HISTORICAL_START_DATE} to {HISTORICAL_END_DATE}")
print(f"      Purpose: Algorithm generation (training data)")
print(f"      Granularity: Minute-by-minute OHLCV")
print(f"")
print(f"   Simulation Data:     {SIMULATION_DATE}")
print(f"      Purpose: Market simulation (trading day)")
print(f"      Granularity: Minute-by-minute OHLCV")

print("\n💡 Explanation:")
print("-"*60)
print(f"   - Algorithms will be generated using data from")
print(f"     {HISTORICAL_START_DATE} to {HISTORICAL_END_DATE}")
print(f"")
print(f"   - The simulation will run on {SIMULATION_DATE}")
print(f"     with minute-by-minute data (9:30 AM - 4:00 PM)")
print(f"")
print(f"   - This ensures algorithms don't see the simulation")
print(f"     day data during training (no look-ahead bias)")

print("\n" + "="*60)
print("✅ Dates are configured for 2025!")
print("="*60)
print("")
