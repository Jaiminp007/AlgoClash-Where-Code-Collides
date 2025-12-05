from __future__ import annotations

import sys
import os
from pathlib import Path
import traceback
import hashlib
import random
import shutil

# Add the open_router directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent / "open_router"))

from market.tick_generator import MongoDBTickGenerator, display_stock_chart
from market.market_simulation import MarketSimulation
from market.agent import MarketMakerAgent
# from generate_stock_data import generate_stock_data_for_ticker  <-- Removed

import warnings
from typing import Callable, Dict, Any, List
import uuid
import importlib.util
import inspect

# Optimized AlgoAgent that loads the module once and calls it directly
class AlgoAgent:
    def __init__(self, name: str, module_path: str, symbol: str):
        self.name = name
        self.symbol = symbol
        self._module_path = module_path
        self._hold_streak = 0  # Count consecutive HOLDs to seed positions
        # Load once; keep original function and its signature for best-effort compatibility
        self._trade_function, self._trade_sig = self._load_module()
        # Store the original algorithm code for adaptation prompts
        self._algorithm_code = self._load_algorithm_code()
        # Per-agent deterministic behavior parameters to diversify actions
        seed_int = int(hashlib.md5(self.name.encode()).hexdigest()[:8], 16)
        self._rng = random.Random(seed_int)
        # Trade frequency throttle: DISABLED - let all agents act every tick for max trades
        self._tick_skip_mod = 1  # Always trade every tick
        # Position sizing: MORE AGGRESSIVE for higher ROI with leverage
        self._buy_frac_lo = self._rng.uniform(0.15, 0.30)
        self._buy_frac_hi = min(0.80, self._buy_frac_lo + self._rng.uniform(0.20, 0.40))
        self._sell_frac = self._rng.uniform(0.30, 0.70)
        # Limit price offsets in basis points (bps) - TIGHTER for faster fills
        self._buy_bps = self._rng.uniform(5.0, 15.0)   # 0.05% - 0.15% above market
        self._sell_bps = self._rng.uniform(5.0, 15.0)  # 0.05% - 0.15% below market
        # HOLD seeding cadence and preference - MORE AGGRESSIVE
        self._seed_ticks = self._rng.randint(1, 3)  # Seed faster
        self._seed_prefer_sell = self._rng.choice([True, False])
        print(f"✅ {self.name}: Loaded with direct import optimization")
    
    def _load_algorithm_code(self) -> str:
        """Load the algorithm source code for adaptation prompts."""
        try:
            with open(self._module_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""

    def _load_module(self) -> tuple[Callable[..., str], inspect.Signature | None]:
        """Load the execute_trade function from the agent's module path and capture its signature."""
        try:
            spec = importlib.util.spec_from_file_location(self.name, self._module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'execute_trade') and callable(module.execute_trade):
                    try:
                        return module.execute_trade, inspect.signature(module.execute_trade)
                    except Exception:
                        # Signature inspection can fail for dynamically generated functions
                        return module.execute_trade, None
        except Exception as e:
            print(f"❌ Error loading module for {self.name}: {e}")
        # Return a dummy function if loading fails
        return (lambda ticker, cash, stock: "HOLD"), None

    def _call_trade(self, symbol: str, price: float, tick: int, cash: float, stock: int) -> str:
        """Invoke execute_trade with best-effort compatibility across signatures.
        Preferred signature: (symbol, price, tick, cash, stock)
        Legacy supported: (symbol, price, cash, stock) or (symbol, cash, stock)
        """
        return self._call_trade_with_fn(self._trade_function, symbol, price, tick, cash, stock)
    
    def _call_trade_with_fn(self, fn, symbol: str, price: float, tick: int, cash: float, stock: int) -> str:
        """Invoke a trade function with best-effort compatibility across signatures."""
        try:
            sig = None
            try:
                sig = inspect.signature(fn)
            except Exception:
                pass
            
            if sig is not None:
                params = list(sig.parameters.keys())
                # Newest: (symbol, price, tick, cash, stock)
                if len(params) >= 5:
                    return fn(symbol, price, tick, cash, stock)
                # Transitional: (symbol, price, cash, stock)
                if len(params) == 4:
                    return fn(symbol, price, cash, stock)
                # Legacy: (symbol, cash, stock)
                return fn(symbol, cash, stock)
            # No signature info; try newest then fallback
            try:
                return fn(symbol, price, tick, cash, stock)
            except TypeError:
                try:
                    return fn(symbol, price, cash, stock)
                except TypeError:
                    return fn(symbol, cash, stock)
        except Exception:
            return "HOLD"

    def on_tick(self, price: float, current_tick: int, cash: float = 0.0, stock: int = 0):
        """Execute the AI-generated trading algorithm with optimized isolation."""
        try:
            # Optional per-agent tick throttling
            if self._tick_skip_mod > 1 and (current_tick % self._tick_skip_mod != 0):
                return []

            # Use hot-swapped execute_trade if available, otherwise use original
            trade_fn = getattr(self, 'execute_trade_fn', None) or self._trade_function
            
            # Execute the AI algorithm's decision function directly (with price/tick awareness when supported)
            decision = self._call_trade_with_fn(trade_fn, self.symbol, price, current_tick, cash, stock)
            
            # Parse decision - can be "BUY", "SELL", "HOLD" or tuple like ("BUY", 10)
            action = decision
            specified_qty = None
            if isinstance(decision, tuple) and len(decision) >= 2:
                action = decision[0]
                specified_qty = int(decision[1]) if decision[1] else None
            
            # Process the AI's decision (un-gated to allow margin/short; engine clamps size)
            if action == "BUY":
                self._hold_streak = 0
                # Use specified quantity or AGGRESSIVE default (use leverage: 100-200 shares)
                qty = specified_qty if specified_qty else self._rng.randint(100, 200)
                bid_price = round(price * (1.0 + self._buy_bps / 10000.0), 2)
                return [{"agent": self.name, "side": "buy", "price": bid_price, "quantity": qty}]

            elif action == "SELL":
                self._hold_streak = 0
                # Use specified quantity or AGGRESSIVE default (use leverage: 100-200 shares)
                qty = specified_qty if specified_qty else self._rng.randint(100, 200)
                ask_price = round(price * (1.0 - self._sell_bps / 10000.0), 2)
                return [{"agent": self.name, "side": "sell", "price": ask_price, "quantity": qty}]

            # If HOLD, track streaks to seed a position if algos are too conservative
            else: # decision == "HOLD"
                self._hold_streak += 1
                if self._hold_streak >= self._seed_ticks:
                    self._hold_streak = 0 # Reset after seeding
                    # Prefer a small sell if configured and we hold stock; otherwise seed a buy
                    if self._seed_prefer_sell and stock > 0:
                        qty = max(5, min(30, stock))  # MORE AGGRESSIVE seeding
                        ask_price = round(price * (1.0 - self._sell_bps / 20000.0), 2)  # half offset
                        return [{"agent": self.name, "side": "sell", "price": ask_price, "quantity": qty}]
                    elif cash > price:
                        max_affordable = int(cash // price)
                        if max_affordable > 0:
                            qty = max(5, min(30, max_affordable))  # MORE AGGRESSIVE seeding
                            bid_price = round(price * (1.0 + self._buy_bps / 20000.0), 2)  # half offset
                            return [{"agent": self.name, "side": "buy", "price": bid_price, "quantity": qty}]
        
        except Exception as e:
            print(f"⚠️ Error executing {self.name} algorithm: {e}")
            # Print full traceback to pinpoint source lines inside generated algorithms
            print(traceback.format_exc())
            
        return []

def run_simulation_with_params(selected_agents, symbol, progress_callback=None):
    """Run simulation with specific agents and stock symbol (API-driven)"""
    print(f"🤖 AI TRADER BATTLEFIELD - Market Simulation ({symbol})")
    print("=" * 50)
    
    # Silence noisy future warnings from pandas/yfinance during the battle
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    if progress_callback:
        progress_callback(35, "Starting algorithm generation...")

    # Generate algorithms for selected agents
    print("\n🧠 STEP 1: Generating Trading Algorithms")
    print("-" * 40)

    try:
        from open_router.algo_gen import generate_algorithms_for_agents
        success = generate_algorithms_for_agents(selected_agents, symbol, progress_callback)
        if not success:
            raise RuntimeError("Algorithm generation failed or incomplete; aborting simulation.")
        print("✅ Algorithm generation completed successfully")
        if progress_callback:
            progress_callback(60, "Algorithms generated, starting market simulation...")
    except Exception as e:
        # Do not continue when generation fails
        if progress_callback:
            progress_callback(50, f"Algorithm generation error: {e}")
        raise
    
    return run_market_simulation(symbol, progress_callback, allowed_models=selected_agents)

def run_market_simulation(symbol, progress_callback=None, allowed_models: list[str] | None = None, tick_callback=None, enable_adaptation: bool = False, adaptation_callback=None):
    """Run the market simulation part
    allowed_models: Optional list of model IDs to include (e.g., 'anthropic/claude-haiku-4.5').
    When provided, only algorithms with matching sanitized filenames will be loaded.
    tick_callback: Optional callback function called on each tick with (tick_num, tick_data, trades)
    enable_adaptation: Whether to enable mid-simulation algorithm adaptation at checkpoints
    adaptation_callback: Async callback for adaptation: (agents_data, checkpoint_num) -> Dict[agent_name, new_code]
    """
    if progress_callback:
        progress_callback(65, "Preparing stock chart...")
    
    # Step 2: Display 30-day stock chart (skip in API mode to avoid GUI issues)
    print("\n📈 STEP 2: Displaying Stock Chart")
    print("-" * 40)
    try:
        if progress_callback:
            print("⏭️ Skipping chart display in API mode to avoid GUI issues")
        else:
            display_stock_chart(symbol, days=30)
    except Exception as e:
        print(f"⚠️ Warning: Chart display failed: {e}")
        print("📝 Continuing without chart...")
    
    if progress_callback:
        progress_callback(70, "Loading trading agents...")
    
    # Step 3: Initialize market simulation
    print("\n🏦 STEP 3: Starting Market Simulation")
    print("-" * 40)
    
    # Create tick generator from MongoDB
    
    tick_gen = MongoDBTickGenerator(symbol=symbol)
    if tick_gen.data:
        print(f"✅ Using MongoDB data for simulation ({len(tick_gen.data)} ticks)")
        tick_src = tick_gen.stream(sleep_seconds=0.25)
    else:
        print("❌ MongoDB data unavailable. Please ensure data is populated in the database.")
        return

    # Discover generated algorithm modules
    base_gen = Path(__file__).resolve().parent / "generate_algo"
    base_open = Path(__file__).resolve().parent / "open_router"
    
    # Sanitize previously generated files that may contain markdown code fences
    def _sanitize_generated_files(path: Path):
        try:
            for pyf in path.glob("generated_algo_*.py"):
                try:
                    txt = pyf.read_text(encoding="utf-8")
                except Exception:
                    continue
                if "```" in txt:
                    # Strip leading/trailing code fences
                    lines = [ln for ln in txt.splitlines() if not ln.strip().startswith("```")]
                    cleaned = "\n".join(lines).strip() + "\n"
                    # Do NOT inject stubs; if execute_trade is missing, let loading fail
                    try:
                        pyf.write_text(cleaned, encoding="utf-8")
                        print(f"🧹 Sanitized code fences in {pyf.name}")
                    except Exception as ie:
                        print(f"⚠️ Failed to sanitize {pyf.name}: {ie}")
        except Exception:
            pass

    _sanitize_generated_files(base_gen)
    _sanitize_generated_files(base_open)

    # Discover any generated_algo_*.py files in both locations
    discovered = list(base_gen.glob("generated_algo_*.py")) + list(base_open.glob("generated_algo_*.py"))

    # If a subset of models is specified, filter to only those files
    allowed_stems = None
    if allowed_models:
        def _sanitize(name: str) -> str:
            return name.replace('/', '_').replace('-', '_').replace(':', '_').replace('.', '_')
        allowed_stems = {f"generated_algo_{_sanitize(m)}" for m in allowed_models}
        print(f"🎯 Filtering algorithms for {len(allowed_models)} models:")
        for m in allowed_models:
            print(f"  - {m} → {_sanitize(m)}")
        print(f"📋 Expected stems: {allowed_stems}")

    # Deduplicate by name
    seen = set()
    algo_modules = []
    for p in discovered:
        if p.name in seen:
            print(f"⏭️ Skipping duplicate: {p.name}")
            continue
        if allowed_stems is not None:
            if p.stem not in allowed_stems:
                print(f"🚫 Filtering out (not in allowed list): {p.stem}")
                continue
            else:
                print(f"✅ Including: {p.stem}")
        seen.add(p.name)
        algo_modules.append(p)

    # Load AI-generated trading agents
    agents = []
    if not algo_modules:
        print(f"❌ No generated algorithms found in {base_gen} or {base_open}.")
        print("Please run the algorithm generation first.")
        return
        
    print(f"🔍 Found {len(algo_modules)} algorithm files:")
    for p in algo_modules:
        print(f"  - {p.name}")
    
    # Load each algorithm as a separate agent
    for p in algo_modules:
        try:
            agent = AlgoAgent(name=p.stem, module_path=str(p), symbol=symbol)
            agents.append(agent)
            print(f"✅ Loaded: {p.stem}")
        except Exception as e:
            print(f"❌ Failed to load {p.name}: {e}")

    if not agents:
        print("❌ No valid trading agents could be loaded. Exiting.")
        return
    
    print(f"📊 Loaded {len(agents)} AI trading agents for simulation")
    
    if progress_callback:
        progress_callback(80, f"Loaded {len(agents)} agents, adding liquidity providers...")

    # Add liquidity providers (excluded from the final leaderboard)
    agents.append(MarketMakerAgent("Liquidity_MM1", spread_bps=8.0, quantity=5))
    agents.append(MarketMakerAgent("Liquidity_MM2", spread_bps=12.0, quantity=3))
    print("➕ Added liquidity providers: Liquidity_MM1, Liquidity_MM2")
    
    if progress_callback:
        progress_callback(85, "Starting market simulation...")

    # Create simulation with ORDER BOOK ENABLED and slower pacing for visibility
    from market.market_simulation import SimulationConfig
    config = SimulationConfig(
        max_ticks=390,  # 390 ticks at 0.25s each = ~97.5 seconds total (full trading day)
        tick_sleep=0.25,  # 250ms between ticks for visibility
        log_trades=True,
        log_orders=False,  # Disable for cleaner logs
        enable_order_book=True,  # ENABLE ORDER BOOK for proper matching
        initial_cash=10000.0,
        initial_stock=0,  # Start with 0 stock - AIs must trade to build positions
        mm_initial_stock=150,  # Market makers need stock to provide liquidity
        # Enable margin and short selling to increase volume/ROE dispersion
        allow_negative_cash=True,
        cash_borrow_limit=40000.0,  # INCREASED: 4x initial cash leverage
        allow_short=True,
        max_short_shares=250,  # INCREASED: Can short 1.5x more shares
        # Expire unfilled limit orders each tick to free reservations
        order_ttl_ticks=1,
        # Mid-simulation adaptation settings
        enable_adaptation=enable_adaptation,
        adaptation_checkpoints=[130, 260],  # Pause at 130 and 260 ticks (3 phases of ~130 each)
        adaptation_callback=adaptation_callback
    )

    sim = MarketSimulation(agents, config, tick_callback=tick_callback)

    # Run the simulation
    try:
        if progress_callback:
            progress_callback(90, "Running market simulation...")
        results = sim.run(ticks=tick_src, max_ticks=390, log=True)  # 390 ticks for full trading day simulation
        if progress_callback:
            progress_callback(95, "Calculating final results...")
    except KeyboardInterrupt:
        print("\n⏹️ Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Simulation error: {e}")

    # Compute final results and declare winner
    print("\n" + "=" * 60)
    print("🏁 BATTLE RESULTS (sorted by ROI)")
    print("=" * 60)

    # Prefer simulation's leaderboard (uses fair baseline set on first tick)
    leaderboard = []
    if isinstance(results, dict) and 'leaderboard' in results:
        leaderboard = [row for row in results['leaderboard'] if not row['name'].startswith("Liquidity_")]
    else:
        # Fallback: compute from sim state if results missing
        leaderboard = []
        for name, pf in sim.portfolio.items():
            if name.startswith("Liquidity_"):
                continue
            initial = getattr(sim.agent_manager, 'initial_values', {}).get(name, 10000.0)
            final_val = pf.cash + pf.stock * max(sim.last_price, 0.0)
            roi_val = 0.0 if initial == 0 else (final_val - initial) / initial
            leaderboard.append({
                'name': name,
                'roi': roi_val,
                'current_value': final_val,
                'initial_value': initial,
                'initial_stock': getattr(sim.agent_manager, 'initial_stocks', {}).get(name, 0),
                'cash': pf.cash,
                'stock': pf.stock,
                'trades': len([t for t in sim.agent_manager.trade_records if t.agent_name == name])
            })

    leaderboard.sort(key=lambda x: x['roi'], reverse=True)

    for row in leaderboard:
        print(f"{row['name']}: ROI={row['roi']*100:+.2f}% | Final=${row['current_value']:.2f} (cash=${row['cash']:.2f}, stock={row['stock']})")

    if leaderboard:
        winner = leaderboard[0]
        print("-" * 60)
        print(f"🏆 Winner: {winner['name']} with ROI {winner['roi']*100:+.2f}% and Final ${winner['current_value']:.2f}")

        # Show performance analysis
        print("\n📊 PERFORMANCE ANALYSIS:")
        print("-" * 30)
        for i, row in enumerate(leaderboard, 1):
            if i == 1:
                print(f"🥇 {row['name']}: {row['roi']*100:+.2f}% ROI")
            elif i == 2:
                print(f"🥈 {row['name']}: {row['roi']*100:+.2f}% ROI")
            elif i == 3:
                print(f"🥉 {row['name']}: {row['roi']*100:+.2f}% ROI")
            else:
                print(f"  {i}. {row['name']}: {row['roi']*100:+.2f}% ROI")
    
    # NOTE: Algorithm cleanup is now handled by explicit API call when user returns to dashboard
    # Algorithms are stored in MongoDB with simulation results before cleanup

    # Return results for API
    return {
        "leaderboard": leaderboard,
        "winner": leaderboard[0] if leaderboard else None,
        "symbol": symbol,
        "total_agents": len(leaderboard)
    }

def main():
    """Original main function for direct execution"""
    symbol = "AAPL"
    return run_market_simulation(symbol)


if __name__ == "__main__":
    main()