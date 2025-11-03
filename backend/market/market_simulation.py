"""
Market Simulation Engine
Main orchestrator for the trading simulation that coordinates agents, order book, and tick data.
"""

import time
from typing import List, Dict, Any, Iterator, Optional
from dataclasses import dataclass
import uuid

from .order_book import OrderBook, Order, OrderSide, OrderType
from .tick_generator import TickData
from .agent import AgentManager, TradingAgent, Portfolio


@dataclass
class SimulationConfig:
    """Configuration for market simulation."""
    max_ticks: int = 60
    tick_sleep: float = 1.0
    log_trades: bool = True
    log_orders: bool = False
    enable_order_book: bool = True
    initial_cash: float = 10000.0
    initial_stock: int = 0
    # Extra inventory for liquidity providers (by name prefix 'Liquidity_')
    mm_initial_stock: int = 100
    # Margin and short selling settings
    allow_negative_cash: bool = False
    cash_borrow_limit: float = 0.0   # Max magnitude of negative cash allowed
    allow_short: bool = False
    max_short_shares: int = 0        # Max magnitude of negative stock allowed
    # Order expiration (in ticks). 0 means orders persist; >0 expire after N ticks
    order_ttl_ticks: int = 0


class MarketSimulation:
    """
    Main market simulation engine that orchestrates the trading environment.
    
    This class coordinates:
    - Tick data from Yahoo Finance
    - Trading agents and their decisions
    - Order book for trade matching
    - Portfolio management and performance tracking
    """
    
    def __init__(self, agents: List[TradingAgent], config: Optional[SimulationConfig] = None, tick_callback=None):
        """
        Initialize the market simulation.

        Args:
            agents: List of trading agents to participate
            config: Simulation configuration
            tick_callback: Optional callback function called on each tick with (tick_num, tick_data, trades)
        """
        self.config = config or SimulationConfig()
        self.agent_manager = AgentManager()
        self.order_book = OrderBook()
        self.tick_callback = tick_callback
        
        # Add all agents to the manager
        for agent in agents:
            self.agent_manager.add_agent(agent, self.config.initial_cash)
            # Optionally seed initial stock to enable early sell-side liquidity
            try:
                if getattr(self.config, "initial_stock", 0) > 0:
                    self.agent_manager.portfolios[agent.name].stock = int(self.config.initial_stock)
                # Provide additional seed inventory for designated liquidity providers
                if str(agent.name).startswith("Liquidity_") and getattr(self.config, "mm_initial_stock", 0) > 0:
                    self.agent_manager.portfolios[agent.name].stock = max(
                        int(self.agent_manager.portfolios[agent.name].stock),
                        int(self.config.mm_initial_stock)
                    )
            except Exception:
                pass
        # Propagate margin/short settings
        self.agent_manager.allow_negative_cash = bool(getattr(self.config, 'allow_negative_cash', False))
        self.agent_manager.cash_borrow_limit = float(getattr(self.config, 'cash_borrow_limit', 0.0))
        self.agent_manager.allow_short = bool(getattr(self.config, 'allow_short', False))
        self.agent_manager.max_short_shares = int(getattr(self.config, 'max_short_shares', 0))
            
        # Simulation state
        self.current_tick = 0
        self.last_price = 0.0
        self.first_price = None
        self.is_running = False
        self.simulation_start_time = 0.0
        
        # Statistics
        self.total_trades = 0
        self.total_volume = 0
        self.tick_history: List[Dict[str, Any]] = []
        
        # Reservation ledger (prevents overspending/overselling for resting orders)
        # Per-order reservations
        self._order_meta: Dict[str, Dict[str, Any]] = {}
        self._order_reserved_cash: Dict[str, float] = {}
        self._order_reserved_stock: Dict[str, int] = {}
        # Aggregated per-agent reservations
        self._agent_reserved_cash: Dict[str, float] = {}
        self._agent_reserved_stock: Dict[str, int] = {}
        
        print(f"🏦 Market Simulation initialized with {len(agents)} agents")
        
    @property
    def portfolio(self) -> Dict[str, Portfolio]:
        """Get current portfolios of all agents."""
        return self.agent_manager.portfolios
        
    def run(self, ticks: Iterator[TickData], max_ticks: Optional[int] = None, log: bool = None) -> Dict[str, Any]:
        """
        Run the market simulation with the provided tick stream.
        
        Args:
            ticks: Iterator providing market tick data
            max_ticks: Maximum number of ticks to process
            log: Whether to log simulation progress
            
        Returns:
            Simulation results and statistics
        """
        max_ticks = max_ticks or self.config.max_ticks
        log = log if log is not None else self.config.log_trades
        
        print("🚀 Starting market simulation...")
        print(f"⚙️ Config: max_ticks={max_ticks}, agents={len(self.agent_manager.agents)}")
        
        self.is_running = True
        self.simulation_start_time = time.time()
        self.current_tick = 0
        
        try:
            for tick_data in ticks:
                if self.current_tick >= max_ticks:
                    print(f"\\n⏱️ Reached maximum ticks ({max_ticks})")
                    break
                    
                self._process_tick(tick_data, log)
                self.current_tick += 1
                
                # Allow for graceful interruption
                if not self.is_running:
                    break
                    
        except KeyboardInterrupt:
            print("\\n⏹️ Simulation interrupted by user")
        except Exception as e:
            print(f"\\n❌ Simulation error: {e}")
        finally:
            self.is_running = False
            
        return self._generate_results()
        
    def _process_tick(self, tick_data: TickData, log: bool = True):
        """Process a single market tick."""
        current_price = tick_data.close
        self.last_price = current_price
        # Initialize fair ROI baselines on the first observed price
        if self.first_price is None:
            self.first_price = current_price
            try:
                for agent_name, pf in self.agent_manager.portfolios.items():
                    self.agent_manager.initial_values[agent_name] = pf.cash + (pf.stock * current_price)
                    self.agent_manager.initial_stocks[agent_name] = pf.stock
                    if log and not agent_name.startswith("Liquidity_"):
                        print(f"📊 {agent_name}: Initial stock={pf.stock}, cash=${pf.cash:.2f}, total=${self.agent_manager.initial_values[agent_name]:.2f}")
            except Exception as e:
                print(f"⚠️ Error initializing baseline values: {e}")
        
        if log and self.current_tick % 10 == 0:
            print(f"\\n⏰ Tick {self.current_tick}: {tick_data.symbol} @ ${current_price:.2f}")
            
        # Get trading decisions from all agents
        agent_orders = self.agent_manager.get_agent_decisions(current_price, self.current_tick)
        
        if self.config.log_orders and agent_orders:
            print(f"📋 Generated {len(agent_orders)} orders")
            
        # Process orders through the order book if enabled
        if self.config.enable_order_book:
            self._process_orders_through_book(agent_orders, log)
        else:
            # Simple execution without order book
            self._execute_orders_simple(agent_orders, current_price, log)
            
        # Record tick data
        self._record_tick_data(tick_data, len(agent_orders))
        # Expire any resting orders at end of tick and release reservations
        self._expire_and_cancel_orders()

        # Call tick callback if provided (for real-time updates to frontend)
        if self.tick_callback:
            try:
                # Get recent trades from this tick
                recent_trades = []
                if self.config.enable_order_book:
                    # Get trades that happened in this tick (last N trades)
                    all_trades = self.order_book.trades
                    if all_trades:
                        # Estimate trades from this tick (simple heuristic)
                        recent_trades = all_trades[-min(20, len(all_trades)):]

                self.tick_callback(
                    self.current_tick,
                    {
                        'price': current_price,
                        'timestamp': tick_data.timestamp,
                        'volume': tick_data.volume
                    },
                    recent_trades
                )
            except Exception as e:
                print(f"⚠️ Tick callback error: {e}")
        
    def _process_orders_through_book(self, orders: List[Dict[str, Any]], log: bool = True):
        """Process orders through the order book for realistic matching."""
        for order_data in orders:
            try:
                agent_name = order_data['agent']
                side_str = order_data['side'].lower()
                qty_requested = int(order_data['quantity'])
                limit_price = float(order_data['price'])
                side = OrderSide.BUY if side_str == 'buy' else OrderSide.SELL
                
                # Capacity check with reservations (existing + per-tick)
                portfolio = self.agent_manager.portfolios.get(agent_name)
                if not portfolio:
                    continue
                
                # Compute available capacity after existing reservations
                reserved_cash_total = self._agent_reserved_cash.get(agent_name, 0.0)
                reserved_stock_total = self._agent_reserved_stock.get(agent_name, 0)
                
                if side == OrderSide.BUY:
                    effective_cash = (portfolio.cash - reserved_cash_total)
                    if self.agent_manager.allow_negative_cash:
                        effective_cash += float(self.agent_manager.cash_borrow_limit)
                    max_affordable = int(max(0.0, effective_cash) // max(limit_price, 1e-9))
                    if max_affordable <= 0:
                        if log and self.config.log_orders:
                            print(f"🚫 {agent_name} BUY skipped: insufficient cash for {qty_requested} @ ${limit_price:.2f}")
                        continue
                    qty = min(qty_requested, max_affordable)
                    if qty <= 0:
                        continue
                else:  # SELL
                    effective_stock = int(portfolio.stock - reserved_stock_total)
                    if self.agent_manager.allow_short:
                        effective_stock += int(self.agent_manager.max_short_shares)
                    available_stock = int(max(0, effective_stock))
                    if available_stock <= 0:
                        # Quietly skip when capacity is exhausted (avoid noisy logs)
                        continue
                    qty = min(qty_requested, available_stock)
                    if qty <= 0:
                        continue
                
                # Create order with adjusted, feasible quantity
                order = Order(
                    order_id=str(uuid.uuid4()),
                    agent_name=agent_name,
                    side=side,
                    quantity=qty,
                    price=limit_price,
                    order_type=OrderType.LIMIT
                )
                
                # Log the order being placed
                if log and self.config.log_orders:
                    side_str = "BUY" if order.side == OrderSide.BUY else "SELL"
                    print(f"📝 {order.agent_name} places {side_str} order: {order.quantity} @ ${order.price:.2f}")
                
                # Reserve capacity for this order (entire limit quantity); will release on fills
                if side == OrderSide.BUY:
                    reserve_amt = order.quantity * order.price
                    self._order_reserved_cash[order.order_id] = reserve_amt
                    self._agent_reserved_cash[agent_name] = self._agent_reserved_cash.get(agent_name, 0.0) + reserve_amt
                else:
                    reserve_qty = order.quantity
                    self._order_reserved_stock[order.order_id] = reserve_qty
                    self._agent_reserved_stock[agent_name] = self._agent_reserved_stock.get(agent_name, 0) + reserve_qty
                # Track order meta
                self._order_meta[order.order_id] = {
                    "agent": agent_name,
                    "side": side.value,
                    "price": order.price,
                    "created_tick": self.current_tick,
                    "ttl": int(getattr(self.config, 'order_ttl_ticks', 0))
                }
                
                # Add to order book and get resulting trades
                trades = self.order_book.add_order(order)
                
                # Execute the trades in agent portfolios
                for trade in trades:
                    # Execute buy side
                    buy_success = self.agent_manager.execute_trade(
                        trade.buy_agent, 'buy', trade.quantity, trade.price
                    )
                    # Execute sell side
                    sell_success = self.agent_manager.execute_trade(
                        trade.sell_agent, 'sell', trade.quantity, trade.price
                    )
                    
                    # Adjust reservations for both matched orders using order IDs
                    # BUY side reservation release
                    if hasattr(trade, 'buy_order_id') and trade.buy_order_id in self._order_reserved_cash:
                        limit_p = self._order_meta.get(trade.buy_order_id, {}).get('price', trade.price)
                        dec_amt = min(self._order_reserved_cash.get(trade.buy_order_id, 0.0), limit_p * trade.quantity)
                        self._order_reserved_cash[trade.buy_order_id] -= dec_amt
                        self._agent_reserved_cash[trade.buy_agent] = max(0.0, self._agent_reserved_cash.get(trade.buy_agent, 0.0) - dec_amt)
                        if self._order_reserved_cash[trade.buy_order_id] <= 1e-9:
                            del self._order_reserved_cash[trade.buy_order_id]
                            self._order_meta.pop(trade.buy_order_id, None)
                    
                    # SELL side reservation release
                    if hasattr(trade, 'sell_order_id') and trade.sell_order_id in self._order_reserved_stock:
                        dec_qty = min(self._order_reserved_stock.get(trade.sell_order_id, 0), trade.quantity)
                        self._order_reserved_stock[trade.sell_order_id] -= dec_qty
                        self._agent_reserved_stock[trade.sell_agent] = max(0, self._agent_reserved_stock.get(trade.sell_agent, 0) - dec_qty)
                        if self._order_reserved_stock[trade.sell_order_id] <= 0:
                            del self._order_reserved_stock[trade.sell_order_id]
                            self._order_meta.pop(trade.sell_order_id, None)
                    
                    if buy_success and sell_success:
                        self.total_trades += 1
                        self.total_volume += trade.quantity
                    
                    if log and self.config.log_trades:
                        print(f"💰 Trade: {trade.buy_agent} bought {trade.quantity} @ ${trade.price:.2f} from {trade.sell_agent}")
                
            except Exception as e:
                if log:
                    print(f"⚠️ Error processing order from {order_data.get('agent', 'Unknown')}: {e}")
                    
        # Log order book state periodically
        if log and self.current_tick % 10 == 0:
            self._log_order_book_state()
        
    def _expire_and_cancel_orders(self):
        """Expire resting orders based on configured TTL and release reservations."""
        ttl_default = int(getattr(self.config, 'order_ttl_ticks', 0))
        if ttl_default <= 0:
            return
        
        # Iterate over a copy since we will mutate structures
        for order_id, meta in list(self._order_meta.items()):
            ttl = int(meta.get('ttl', ttl_default))
            created_tick = int(meta.get('created_tick', self.current_tick))
            if ttl > 0 and (self.current_tick - created_tick) >= ttl:
                # Cancel order from book (if still present)
                try:
                    self.order_book.cancel_order(order_id)
                except Exception:
                    pass
                agent = meta.get('agent', '')
                side = meta.get('side', '')
                # Release reservations
                if side == 'buy' and order_id in self._order_reserved_cash:
                    dec_amt = self._order_reserved_cash.pop(order_id)
                    self._agent_reserved_cash[agent] = max(0.0, self._agent_reserved_cash.get(agent, 0.0) - dec_amt)
                elif side == 'sell' and order_id in self._order_reserved_stock:
                    dec_qty = self._order_reserved_stock.pop(order_id)
                    self._agent_reserved_stock[agent] = max(0, self._agent_reserved_stock.get(agent, 0) - dec_qty)
                # Remove meta
                self._order_meta.pop(order_id, None)
                    
    def _log_order_book_state(self):
        """Log the current state of the order book."""
        print(f"\n📊 Order Book State (Tick {self.current_tick}):")
        print(f"   Best Bid: ${self.order_book.best_bid:.2f}" if self.order_book.best_bid else "   Best Bid: None")
        print(f"   Best Ask: ${self.order_book.best_ask:.2f}" if self.order_book.best_ask else "   Best Ask: None")
        print(f"   Total Orders: {len(self.order_book.orders)}")
        print(f"   Total Trades: {len(self.order_book.trades)}")
        
        # Show some sample orders
        if self.order_book.orders:
            print("   Sample Orders:")
            count = 0
            for order_id, order in self.order_book.orders.items():
                if count >= 3:  # Show only first 3 orders
                    break
                side_str = "BUY" if order.side == OrderSide.BUY else "SELL"
                print(f"     {order.agent_name}: {side_str} {order.quantity} @ ${order.price:.2f}")
                count += 1
                    
    def _execute_orders_simple(self, orders: List[Dict[str, Any]], current_price: float, log: bool = True):
        """Simple order execution without order book matching."""
        for order in orders:
            try:
                agent_name = order['agent']
                side = order['side'].lower()
                quantity = order['quantity']
                price = current_price  # Use current market price
                
                success = self.agent_manager.execute_trade(agent_name, side, quantity, price)
                
                if success:
                    self.total_trades += 1
                    self.total_volume += quantity
                    
                    if log and self.config.log_trades:
                        print(f"💰 {agent_name} {side} {quantity} @ ${price:.2f}")
                        
            except Exception as e:
                if log:
                    print(f"⚠️ Error executing order: {e}")
                    
    def _record_tick_data(self, tick_data: TickData, order_count: int):
        """Record tick data for analysis."""
        self.tick_history.append({
            'tick': self.current_tick,
            'timestamp': tick_data.timestamp,
            'price': tick_data.close,
            'volume': tick_data.volume,
            'orders': order_count,
            'trades': len(self.order_book.trades) if self.config.enable_order_book else self.total_trades
        })
        
    def _generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive simulation results."""
        simulation_time = time.time() - self.simulation_start_time
        leaderboard = self.agent_manager.get_leaderboard(self.last_price)
        
        results = {
            'simulation_stats': {
                'total_ticks': self.current_tick,
                'simulation_time_seconds': simulation_time,
                'total_trades': self.total_trades,
                'total_volume': self.total_volume,
                'final_price': self.last_price,
                'initial_price': self.first_price,
                'trades_per_tick': self.total_trades / max(self.current_tick, 1)
            },
            'leaderboard': leaderboard,
            'order_book_stats': self.order_book.get_stats() if self.config.enable_order_book else {},
            'tick_history': self.tick_history
        }
        
        return results
        
    def stop(self):
        """Stop the simulation."""
        self.is_running = False
        
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time statistics during simulation."""
        return {
            'current_tick': self.current_tick,
            'last_price': self.last_price,
            'total_trades': self.total_trades,
            'total_volume': self.total_volume,
            'top_3_agents': self.agent_manager.get_leaderboard(self.last_price)[:3],
            'order_book': {
                'best_bid': self.order_book.best_bid,
                'best_ask': self.order_book.best_ask,
                'spread': (self.order_book.best_ask - self.order_book.best_bid) 
                         if self.order_book.best_bid and self.order_book.best_ask else None
            } if self.config.enable_order_book else {}
        }
        
    def display_live_leaderboard(self):
        """Display live leaderboard during simulation."""
        if self.current_tick % 10 == 0 and self.current_tick > 0:
            print("\\n" + "="*60)
            print(f"📊 LIVE LEADERBOARD - Tick {self.current_tick}")
            print("="*60)
            
            leaderboard = self.agent_manager.get_leaderboard(self.last_price)[:5]
            for i, result in enumerate(leaderboard, 1):
                print(f"{i}. {result['name']}: ROI={result['roi']*100:+.2f}% | "
                      f"Value=${result['current_value']:.2f} | Trades={result['trades']}")
                      
            print("="*60)
            
    def add_agent_during_simulation(self, agent: TradingAgent) -> bool:
        """Add an agent during simulation (hot-plugging)."""
        try:
            self.agent_manager.add_agent(agent, self.config.initial_cash)
            print(f"🔥 Hot-plugged new agent: {agent.name}")
            return True
        except Exception as e:
            print(f"❌ Failed to add agent {agent.name}: {e}")
            return False
            
    def remove_agent_during_simulation(self, agent_name: str) -> bool:
        """Remove an agent during simulation."""
        success = self.agent_manager.remove_agent(agent_name)
        if success:
            print(f"🔌 Unplugged agent: {agent_name}")
        return success
        
    def get_market_depth(self, levels: int = 5) -> Dict[str, Any]:
        """Get current market depth from order book."""
        if self.config.enable_order_book:
            return self.order_book.get_market_depth(levels)
        return {"error": "Order book disabled"}
        
    def export_results(self, filename: Optional[str] = None) -> str:
        """Export simulation results to JSON file."""
        import json
        from datetime import datetime
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results_{timestamp}.json"
            
        results = self._generate_results()
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"📁 Results exported to: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Failed to export results: {e}")
            return ""


# Helper function to create a basic simulation
def create_basic_simulation(agents: List[TradingAgent], 
                          max_ticks: int = 60, 
                          enable_order_book: bool = True) -> MarketSimulation:
    """Create a basic market simulation with sensible defaults."""
    config = SimulationConfig(
        max_ticks=max_ticks,
        tick_sleep=1.0,
        log_trades=True,
        log_orders=False,
        enable_order_book=enable_order_book,
        initial_cash=10000.0
    )
    
    return MarketSimulation(agents, config)


if __name__ == "__main__":
    # Test the market simulation
    print("🧪 Testing Market Simulation Engine")
    
    from .agent import create_sample_agents
    from .tick_generator import YFinanceTickGenerator
    
    # Create sample agents
    agents = create_sample_agents()
    
    # Create simulation
    sim = create_basic_simulation(agents, max_ticks=20)
    
    # Create tick generator
    tick_gen = YFinanceTickGenerator("AAPL", period="1d", interval="1m")
    
    print("\\n🚀 Starting test simulation...")
    results = sim.run(tick_gen.stream(sleep_seconds=0.1), max_ticks=20)
    
    print("\\n" + "="*60)
    print("📊 SIMULATION COMPLETE")
    print("="*60)
    
    # Display results
    stats = results['simulation_stats']
    print(f"Total Ticks: {stats['total_ticks']}")
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Total Volume: {stats['total_volume']:,}")
    print(f"Final Price: ${stats['final_price']:.2f}")
    
    print("\\n🏆 FINAL LEADERBOARD:")
    for i, result in enumerate(results['leaderboard'][:5], 1):
        print(f"{i}. {result['name']}: ROI={result['roi']*100:+.2f}% | "
              f"Value=${result['current_value']:.2f}")
