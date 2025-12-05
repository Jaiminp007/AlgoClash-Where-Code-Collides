"""
Market Simulation Engine
Main orchestrator for the trading simulation that coordinates agents, order book, and tick data.
"""

import time
from typing import List, Dict, Any, Iterator, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid

# Database helpers for Mongo-backed tick streams
try:
    from database import get_db
except ImportError:
    try:
        from ..database import get_db  # type: ignore
    except ImportError:
        get_db = None  # Fallback when running standalone

from .order_book import OrderBook, Order, OrderSide, OrderType
from .tick_generator import TickData
from .agent import AgentManager, TradingAgent, Portfolio


@dataclass
class SimulationConfig:
    """Configuration for market simulation."""
    max_ticks: int = 250
    tick_sleep: float = 0.25
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
    # Mid-simulation adaptation settings
    enable_adaptation: bool = False  # Whether to allow algorithm adaptation at checkpoints
    adaptation_checkpoints: List[int] = None  # Tick numbers to pause for adaptation (default: [130, 260])
    adaptation_callback: Any = None  # Async callback for adaptation: (agent_name, performance_data) -> Optional[new_code]


class MongoDBTickGenerator:
    """Stream tick data from MongoDB collections like AAPL_simulation."""

    def __init__(self, ticker: str):
        if get_db is None:
            raise RuntimeError("MongoDB connection helper get_db is unavailable")
        self.ticker = ticker.upper()
        self.db = get_db()
        if self.db is None:
            raise RuntimeError("Failed to connect to MongoDB")
        self.collection_name = f"{self.ticker}_simulation"
        self.collection = self.db[self.collection_name]

    def stream(self, sleep_seconds: float = 0.0) -> Iterator[TickData]:
        """Yield TickData rows ordered by timestamp."""
        cursor = self.collection.find().sort("datetime", 1)
        total = self.collection.count_documents({})
        print(
            f"📈 Loading {self.ticker} data from MongoDB collection '{self.collection_name}' ({total} records)..."
        )
        if total == 0:
            print(f"⚠️ Warning: No simulation data found for {self.ticker}")

        for doc in cursor:
            dt_raw = doc.get("datetime")
            if isinstance(dt_raw, str):
                try:
                    timestamp = datetime.strptime(dt_raw, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    timestamp = datetime.fromisoformat(dt_raw)
            else:
                timestamp = dt_raw

            tick = TickData(
                symbol=doc.get("ticker", self.ticker),
                timestamp=timestamp,
                open_price=float(doc.get("open", 0.0)),
                high=float(doc.get("high", 0.0)),
                low=float(doc.get("low", 0.0)),
                close=float(doc.get("close", 0.0)),
                volume=int(doc.get("volume", 0)),
            )

            yield tick

            if sleep_seconds:
                time.sleep(sleep_seconds)


class MarketSimulation:
    """
    Main market simulation engine that orchestrates the trading environment.
    
    This class coordinates:
    - Tick data from Yahoo Finance
    - Trading agents and their decisions
    - Order book for trade matching
    - Portfolio management and performance tracking
    - Mid-simulation algorithm adaptation at checkpoints
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
        
        # Set default adaptation checkpoints if not specified
        if self.config.adaptation_checkpoints is None:
            self.config.adaptation_checkpoints = [130, 260]  # Pause at 130 and 260 ticks (3 phases of 130 each)
        
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
        
        # Price history for adaptation prompts
        self.price_history: List[float] = []
        
        # Track adaptation events
        self.adaptation_results: Dict[str, List[Dict[str, Any]]] = {}  # agent_name -> list of adaptation events
        
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
        if self.config.enable_adaptation:
            print(f"🔄 Adaptation enabled at checkpoints: {self.config.adaptation_checkpoints}")
        
    @property
    def portfolio(self) -> Dict[str, Portfolio]:
        """Get current portfolios of all agents."""
        return self.agent_manager.portfolios
        
    def run(
        self,
        ticks: Optional[Iterator[TickData]] = None,
        max_ticks: Optional[int] = None,
        log: bool = None,
        ticker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the market simulation with the provided tick stream.
        
        Args:
            ticks: Iterator providing market tick data
            max_ticks: Maximum number of ticks to process
            log: Whether to log simulation progress
            
        Returns:
            Simulation results and statistics
        """
        if ticks is None:
            if not ticker:
                raise ValueError("Simulation requires a tick iterator or a ticker for MongoDB streaming")
            generator = MongoDBTickGenerator(ticker)
            ticks = generator.stream(sleep_seconds=self.config.tick_sleep)

        max_ticks = max_ticks or self.config.max_ticks
        log = log if log is not None else self.config.log_trades
        
        print("🚀 Starting market simulation...")
        print(f"⚙️ Config: max_ticks={max_ticks}, agents={len(self.agent_manager.agents)}")
        if self.config.enable_adaptation:
            print(f"🔄 Adaptation checkpoints: {self.config.adaptation_checkpoints}")
        
        self.is_running = True
        self.simulation_start_time = time.time()
        self.current_tick = 0
        
        # Track which checkpoints we've hit
        checkpoints_completed = set()
        
        try:
            for tick_data in ticks:
                if self.current_tick >= max_ticks:
                    print(f"\\n⏱️ Reached maximum ticks ({max_ticks})")
                    break
                    
                self._process_tick(tick_data, log)
                self.current_tick += 1
                
                # Check for adaptation checkpoint
                if self.config.enable_adaptation and self.config.adaptation_callback:
                    for checkpoint in self.config.adaptation_checkpoints:
                        if self.current_tick == checkpoint and checkpoint not in checkpoints_completed:
                            checkpoints_completed.add(checkpoint)
                            checkpoint_num = self.config.adaptation_checkpoints.index(checkpoint) + 1
                            print(f"\\n{'='*60}")
                            print(f"🔄 ADAPTATION CHECKPOINT {checkpoint_num} at tick {checkpoint}")
                            print(f"{'='*60}")
                            
                            # Run adaptation for all agents
                            self._run_adaptation_checkpoint(
                                checkpoint_num=checkpoint_num,
                                ticker=ticker or tick_data.symbol,
                                total_ticks=max_ticks,
                                log=log
                            )
                            
                            print(f"{'='*60}")
                            print(f"▶️ Resuming simulation...")
                            print(f"{'='*60}\\n")
                
                # Allow for graceful interruption
                if not self.is_running:
                    break
                    
        except KeyboardInterrupt:
            print("\\n⏹️ Simulation interrupted by user")
        except Exception as e:
            print(f"\\n❌ Simulation error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            
        return self._generate_results()
        
    def _process_tick(self, tick_data: TickData, log: bool = True):
        """Process a single market tick."""
        current_price = tick_data.close
        self.last_price = current_price
        
        # Track price history for adaptation prompts
        self.price_history.append(current_price)
        if len(self.price_history) > 100:  # Keep last 100 prices
            self.price_history.pop(0)
        
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

        # Check for margin calls and liquidate positions if necessary
        self._check_and_handle_margin_calls(current_price, log)

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

                # Calculate portfolio values for each agent (cash + stock value)
                agent_portfolios = {}
                for agent_name, portfolio in self.agent_manager.portfolios.items():
                    # Skip liquidity providers from chart
                    if not agent_name.startswith('Liquidity_'):
                        portfolio_value = portfolio.cash + (portfolio.stock * current_price)
                        # Clean up agent name (remove generated_algo_ prefix)
                        clean_name = agent_name.replace('generated_algo_', '')
                        agent_portfolios[clean_name] = {
                            'value': portfolio_value,
                            'cash': portfolio.cash,
                            'stock': portfolio.stock
                        }

                self.tick_callback(
                    self.current_tick,
                    {
                        'price': current_price,
                        'timestamp': tick_data.timestamp,
                        'volume': tick_data.volume,
                        'agent_portfolios': agent_portfolios  # Add agent portfolio values
                    },
                    recent_trades
                )
            except Exception as e:
                print(f"⚠️ Tick callback error: {e}")
    
    def _run_adaptation_checkpoint(
        self,
        checkpoint_num: int,
        ticker: str,
        total_ticks: int,
        log: bool = True
    ):
        """
        Run adaptation checkpoint for all agents.
        
        At each checkpoint, each agent is given:
        - Their current ROI
        - Their current algorithm code
        - Market data and price history
        
        They can then choose to keep their algorithm or improve it.
        """
        import asyncio
        
        if not self.config.adaptation_callback:
            return
            
        current_price = self.last_price
        
        # Collect performance data for each agent
        adaptation_tasks = []
        
        for agent_name, agent in self.agent_manager.agents.items():
            # Skip liquidity providers
            if agent_name.startswith("Liquidity_"):
                continue
                
            portfolio = self.agent_manager.portfolios[agent_name]
            initial_value = self.agent_manager.initial_values.get(agent_name, 10000.0)
            current_value = portfolio.get_total_value(current_price)
            roi = (current_value - initial_value) / initial_value if initial_value > 0 else 0.0
            
            # Get agent's current algorithm code (if available)
            current_code = getattr(agent, '_algorithm_code', None) or ""
            
            # Get agent's trade history for adaptation analysis
            agent_trades = []
            for trade in getattr(agent, 'trade_history', []):
                agent_trades.append({
                    'tick': getattr(trade, 'tick', 0),
                    'action': 'BUY' if getattr(trade, 'side', '') == 'buy' else 'SELL',
                    'shares': getattr(trade, 'quantity', 0),
                    'price': getattr(trade, 'price', 0),
                    'pnl': getattr(trade, 'pnl', 0) if hasattr(trade, 'pnl') else 0
                })
            
            performance_data = {
                'agent_name': agent_name,
                'current_roi': roi,
                'current_cash': portfolio.cash,
                'current_shares': portfolio.stock,
                'current_tick': self.current_tick,
                'total_ticks': total_ticks,
                'price_history': self.price_history[-100:],  # Last 100 prices for better analysis
                'checkpoint_num': checkpoint_num,
                'current_code': current_code,
                'ticker': ticker,
                'trades': agent_trades  # Include trade history for detailed analysis
            }
            
            if log:
                roi_pct = roi * 100
                status = "✅" if roi_pct >= 0 else "❌"
                print(f"  {status} {agent_name}: ROI={roi_pct:+.2f}%, Cash=${portfolio.cash:.2f}, Shares={portfolio.stock}")
            
            adaptation_tasks.append(performance_data)
        
        # Call the adaptation callback (runs async)
        if adaptation_tasks:
            print(f"\\n📤 Sending {len(adaptation_tasks)} agents for adaptation analysis...")
            
            try:
                # Run the adaptation callback
                # This is expected to be an async function that returns a dict of agent_name -> new_code
                results = self.config.adaptation_callback(adaptation_tasks, checkpoint_num)
                
                # If it's a coroutine, run it
                if asyncio.iscoroutine(results):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        results = loop.run_until_complete(results)
                    finally:
                        loop.close()
                
                # Process adaptation results
                if results:
                    for agent_name, new_code in results.items():
                        if new_code:
                            # Hot-swap the algorithm
                            success = self._hot_swap_algorithm(agent_name, new_code, log)
                            
                            # Record the adaptation event
                            if agent_name not in self.adaptation_results:
                                self.adaptation_results[agent_name] = []
                            
                            self.adaptation_results[agent_name].append({
                                'checkpoint': checkpoint_num,
                                'tick': self.current_tick,
                                'updated': success,
                                'roi_before': next((t['current_roi'] for t in adaptation_tasks if t['agent_name'] == agent_name), 0)
                            })
                            
                            if success and log:
                                print(f"  🔄 {agent_name}: Algorithm UPDATED")
                        else:
                            if log:
                                print(f"  ➡️ {agent_name}: Keeping current algorithm")
                                
            except Exception as e:
                print(f"❌ Adaptation error: {e}")
                import traceback
                traceback.print_exc()
    
    def _hot_swap_algorithm(self, agent_name: str, new_code: str, log: bool = True) -> bool:
        """
        Hot-swap an agent's algorithm code during simulation.
        
        This creates a new execute_trade function from the new code
        and replaces the agent's on_tick behavior.
        """
        try:
            if agent_name not in self.agent_manager.agents:
                return False
                
            agent = self.agent_manager.agents[agent_name]
            
            # Compile and execute the new code to extract execute_trade
            namespace = {'__name__': f'adapted_algo_{agent_name}'}
            
            # Add numpy if needed
            try:
                import numpy as np
                namespace['np'] = np
                namespace['numpy'] = np
            except ImportError:
                pass
            
            exec(new_code, namespace)
            
            if 'execute_trade' not in namespace:
                if log:
                    print(f"  ⚠️ {agent_name}: New code missing execute_trade function")
                return False
            
            new_execute_trade = namespace['execute_trade']
            
            # Store the new code on the agent
            agent._algorithm_code = new_code
            agent._execute_trade = new_execute_trade
            
            # If the agent has a custom on_tick that wraps execute_trade, it should pick up the new function
            # For GeneratedAlgorithmAgent, we need to update its execute function
            if hasattr(agent, 'execute_trade_fn'):
                agent.execute_trade_fn = new_execute_trade
            
            if log:
                print(f"  ✅ {agent_name}: Successfully swapped algorithm")
            
            return True
            
        except Exception as e:
            if log:
                print(f"  ❌ {agent_name}: Hot-swap failed: {e}")
            return False
        
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

    def _check_and_handle_margin_calls(self, current_price: float, log: bool = True):
        """
        Check all agents for margin calls and liquidate positions if necessary.
        Margin call triggered when portfolio value drops below 50% of initial value.
        """
        margin_threshold = 0.5  # 50% of initial value

        for agent_name in list(self.agent_manager.agents.keys()):
            # Skip liquidity providers
            if agent_name.startswith("Liquidity_"):
                continue

            # Check if agent is in margin call
            if self.agent_manager.check_margin_call(agent_name, current_price, margin_threshold):
                portfolio = self.agent_manager.portfolios[agent_name]
                initial_value = self.agent_manager.initial_values.get(agent_name, 10000.0)
                current_value = portfolio.get_total_value(current_price)

                if log:
                    print(f"\n🚨 MARGIN CALL: {agent_name}")
                    print(f"   Initial Value: ${initial_value:.2f}")
                    print(f"   Current Value: ${current_value:.2f}")
                    print(f"   Loss: ${current_value - initial_value:.2f} ({((current_value/initial_value - 1) * 100):+.2f}%)")
                    print(f"   Cash: ${portfolio.cash:.2f}, Shares: {portfolio.stock}")

                # Liquidate the position
                self.agent_manager.liquidate_position(agent_name, current_price)

                # Update final values after liquidation
                portfolio_after = self.agent_manager.portfolios[agent_name]
                final_value = portfolio_after.get_total_value(current_price)

                if log:
                    print(f"   After Liquidation: ${final_value:.2f} (Cash: ${portfolio_after.cash:.2f}, Shares: {portfolio_after.stock})")

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
            'tick_history': self.tick_history,
            'adaptation_enabled': self.config.enable_adaptation,
            'adaptation_results': self.adaptation_results if self.config.enable_adaptation else {}
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
                          max_ticks: int = 250,
                          enable_order_book: bool = True) -> MarketSimulation:
    """Create a basic market simulation with sensible defaults."""
    config = SimulationConfig(
        max_ticks=max_ticks,
        tick_sleep=0.25,
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
