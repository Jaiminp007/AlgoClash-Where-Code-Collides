"""
Trading Agent Management System
Handles agent portfolios, performance tracking, and trade execution.
"""

from typing import Dict, List, Optional, Protocol, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import uuid


@dataclass
class Portfolio:
    """Represents an agent's portfolio with cash and stock holdings."""
    cash: float = 10000.0  # Starting cash
    stock: int = 0         # Number of shares held
    
    def get_total_value(self, current_price: float) -> float:
        """Calculate total portfolio value at current market price."""
        return self.cash + (self.stock * current_price)
        
    def get_roi(self, initial_value: float, current_price: float) -> float:
        """Calculate Return on Investment as decimal ratio (e.g., 0.5 = 50% gain)."""
        if initial_value <= 0:
            return 0.0
        current_value = self.get_total_value(current_price)
        return (current_value - initial_value) / initial_value


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    trade_id: str
    agent_name: str
    side: str  # 'buy' or 'sell'
    quantity: int
    price: float
    timestamp: float
    portfolio_before: Portfolio
    portfolio_after: Portfolio


class TradingAgent(Protocol):
    """Protocol defining the interface for trading agents."""
    
    name: str
    
    def on_tick(self, price: float, current_tick: int, cash: float = 0.0, stock: int = 0) -> List[Dict[str, Any]]:
        """
        Called on each market tick to generate trading decisions.
        
        Args:
            price: Current market price
            current_tick: Current tick number
            cash: Current cash balance
            stock: Current stock holdings
            
        Returns:
            List of order dictionaries with keys: agent, side, price, quantity
        """
        ...


class BaseAgent(ABC):
    """Base class for trading agents with common functionality."""
    
    def __init__(self, name: str, initial_cash: float = 10000.0):
        self.name = name
        self.initial_cash = initial_cash
        self.trade_history: List[TradeRecord] = []
        self.tick_count = 0
        
    @abstractmethod
    def on_tick(self, price: float, current_tick: int, cash: float = 0.0, stock: int = 0) -> List[Dict[str, Any]]:
        """Implement trading strategy logic."""
        pass
        
    def record_trade(self, trade_record: TradeRecord):
        """Record a trade in the agent's history."""
        self.trade_history.append(trade_record)
        
    def get_performance_stats(self, current_price: float) -> Dict[str, Any]:
        """Get performance statistics for this agent."""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "avg_buy_price": 0.0,
                "avg_sell_price": 0.0,
                "total_volume": 0
            }
            
        buy_trades = [t for t in self.trade_history if t.side == 'buy']
        sell_trades = [t for t in self.trade_history if t.side == 'sell']
        
        return {
            "total_trades": len(self.trade_history),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "avg_buy_price": sum(t.price for t in buy_trades) / len(buy_trades) if buy_trades else 0.0,
            "avg_sell_price": sum(t.price for t in sell_trades) / len(sell_trades) if sell_trades else 0.0,
            "total_volume": sum(t.quantity for t in self.trade_history),
            "total_buy_volume": sum(t.quantity for t in buy_trades),
            "total_sell_volume": sum(t.quantity for t in sell_trades)
        }


class RandomAgent(BaseAgent):
    """Simple random trading agent for baseline comparison."""
    
    def __init__(self, name: str = "RandomAgent", aggressiveness: float = 0.3):
        super().__init__(name)
        self.aggressiveness = aggressiveness  # Probability of making a trade
        
    def on_tick(self, price: float, current_tick: int, cash: float = 0.0, stock: int = 0) -> List[Dict[str, Any]]:
        import random
        
        # Randomly decide whether to trade
        if random.random() > self.aggressiveness:
            return []
            
        # Randomly choose action
        action = random.choice(['buy', 'sell', 'hold'])
        
        if action == 'hold':
            return []
        elif action == 'buy' and cash > price:
            # Buy between 1-3 shares if we have cash
            max_shares = min(3, int(cash // price))
            if max_shares > 0:
                quantity = random.randint(1, max_shares)
                return [{"agent": self.name, "side": "buy", "price": price, "quantity": quantity}]
        elif action == 'sell' and stock > 0:
            # Sell between 1-3 shares if we have stock
            quantity = random.randint(1, min(3, stock))
            return [{"agent": self.name, "side": "sell", "price": price, "quantity": quantity}]
            
        return []


class MomentumAgent(BaseAgent):
    """Momentum-based trading agent that follows price trends."""
    
    def __init__(self, name: str = "MomentumAgent", lookback_period: int = 5):
        super().__init__(name)
        self.lookback_period = lookback_period
        self.price_history: List[float] = []
        
    def on_tick(self, price: float, current_tick: int, cash: float = 0.0, stock: int = 0) -> List[Dict[str, Any]]:
        self.price_history.append(price)
        
        # Keep only the last lookback_period prices
        if len(self.price_history) > self.lookback_period:
            self.price_history.pop(0)
            
        # Need at least 3 prices to determine trend
        if len(self.price_history) < 3:
            return []
            
        # Calculate momentum
        recent_prices = self.price_history[-3:]
        if all(recent_prices[i] > recent_prices[i-1] for i in range(1, len(recent_prices))):
            # Upward momentum - buy
            if cash > price:
                quantity = min(2, int(cash // price))
                if quantity > 0:
                    return [{"agent": self.name, "side": "buy", "price": price, "quantity": quantity}]
        elif all(recent_prices[i] < recent_prices[i-1] for i in range(1, len(recent_prices))):
            # Downward momentum - sell
            if stock > 0:
                quantity = min(2, stock)
                return [{"agent": self.name, "side": "sell", "price": price, "quantity": quantity}]
                
        return []


class MeanReversionAgent(BaseAgent):
    """Mean reversion agent that trades against extreme price movements."""
    
    def __init__(self, name: str = "MeanReversionAgent", window_size: int = 10, threshold: float = 0.02):
        super().__init__(name)
        self.window_size = window_size
        self.threshold = threshold  # 2% threshold
        self.price_history: List[float] = []
        
    def on_tick(self, price: float, current_tick: int, cash: float = 0.0, stock: int = 0) -> List[Dict[str, Any]]:
        self.price_history.append(price)
        
        # Keep only the last window_size prices
        if len(self.price_history) > self.window_size:
            self.price_history.pop(0)
            
        # Need sufficient history to calculate mean
        if len(self.price_history) < self.window_size:
            return []
            
        # Calculate mean and deviation
        if not self.price_history:
            return []
        mean_price = sum(self.price_history) / len(self.price_history)
        deviation = (price - mean_price) / mean_price
        
        if deviation > self.threshold:
            # Price is too high - sell
            if stock > 0:
                quantity = min(2, stock)
                return [{"agent": self.name, "side": "sell", "price": price, "quantity": quantity}]
        elif deviation < -self.threshold:
            # Price is too low - buy
            if cash > price:
                quantity = min(2, int(cash // price))
                if quantity > 0:
                    return [{"agent": self.name, "side": "buy", "price": price, "quantity": quantity}]
                    
        return []


class MarketMakerAgent(BaseAgent):
    """Market maker agent that provides liquidity by placing bid and ask orders."""
    
    def __init__(self, name: str = "MarketMaker", spread_bps: float = 10.0, quantity: int = 1):
        super().__init__(name)
        self.spread_bps = spread_bps  # Spread in basis points
        self.quantity = quantity
        
    def on_tick(self, price: float, current_tick: int, cash: float = 0.0, stock: int = 0) -> List[Dict[str, Any]]:
        orders = []
        
        # Calculate bid-ask spread
        spread = price * (self.spread_bps / 10000.0)
        bid_price = max(price - spread/2, 0.01)
        ask_price = price + spread/2
        
        # Always place both bid and ask; engine will clamp based on capacity limits
        orders.append({
            "agent": self.name,
            "side": "buy", 
            "price": round(bid_price, 2),
            "quantity": self.quantity
        })
        orders.append({
            "agent": self.name,
            "side": "sell",
            "price": round(ask_price, 2), 
            "quantity": self.quantity
        })
        
        return orders


class AgentManager:
    """Manages multiple trading agents and their portfolios."""
    
    def __init__(self):
        self.agents: Dict[str, TradingAgent] = {}
        self.portfolios: Dict[str, Portfolio] = {}
        self.initial_values: Dict[str, float] = {}
        self.initial_stocks: Dict[str, int] = {}
        self.trade_records: List[TradeRecord] = []
        # Margin/short selling settings (can be configured by simulation)
        self.allow_negative_cash: bool = False
        self.cash_borrow_limit: float = 0.0  # Max allowed negative cash (absolute value)
        self.allow_short: bool = False
        self.max_short_shares: int = 0
        
    def add_agent(self, agent: TradingAgent, initial_cash: float = 10000.0):
        """Add a trading agent to the manager."""
        self.agents[agent.name] = agent
        self.portfolios[agent.name] = Portfolio(cash=initial_cash)
        self.initial_values[agent.name] = initial_cash
        
    def remove_agent(self, agent_name: str) -> bool:
        """Remove an agent from the manager."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            del self.portfolios[agent_name]
            del self.initial_values[agent_name]
            return True
        return False
        
    def get_agent_decisions(self, price: float, current_tick: int) -> List[Dict[str, Any]]:
        """Get trading decisions from all agents."""
        all_orders = []
        
        for agent_name, agent in self.agents.items():
            portfolio = self.portfolios[agent_name]
            try:
                orders = agent.on_tick(price, current_tick, portfolio.cash, portfolio.stock)
                if orders:
                    all_orders.extend(orders)
            except Exception as e:
                print(f"⚠️ Error getting decisions from {agent_name}: {e}")
                
        return all_orders
        
    def execute_trade(self, agent_name: str, side: str, quantity: int, price: float) -> bool:
        """Execute a trade and update the agent's portfolio."""
        if agent_name not in self.portfolios:
            return False
            
        portfolio = self.portfolios[agent_name]
        portfolio_before = Portfolio(cash=portfolio.cash, stock=portfolio.stock)
        
        if side.lower() == 'buy':
            cost = quantity * price
            # Check cash capacity with margin allowance
            if (self.allow_negative_cash and (portfolio.cash - cost) >= -float(self.cash_borrow_limit)) or (portfolio.cash >= cost):
                portfolio.cash -= cost
                portfolio.stock += quantity
                
                # Record the trade
                trade_record = TradeRecord(
                    trade_id=str(uuid.uuid4()),
                    agent_name=agent_name,
                    side='buy',
                    quantity=quantity,
                    price=price,
                    timestamp=time.time(),
                    portfolio_before=portfolio_before,
                    portfolio_after=Portfolio(cash=portfolio.cash, stock=portfolio.stock)
                )
                self.trade_records.append(trade_record)
                
                # Record in agent's history if it's a BaseAgent
                if isinstance(self.agents[agent_name], BaseAgent):
                    self.agents[agent_name].record_trade(trade_record)
                    
                return True
        elif side.lower() == 'sell':
            # Check stock capacity with short-selling allowance
            if (self.allow_short and (portfolio.stock - quantity) >= -int(self.max_short_shares)) or (portfolio.stock >= quantity):
                revenue = quantity * price
                portfolio.cash += revenue
                portfolio.stock -= quantity
                
                # Record the trade
                trade_record = TradeRecord(
                    trade_id=str(uuid.uuid4()),
                    agent_name=agent_name,
                    side='sell',
                    quantity=quantity,
                    price=price,
                    timestamp=time.time(),
                    portfolio_before=portfolio_before,
                    portfolio_after=Portfolio(cash=portfolio.cash, stock=portfolio.stock)
                )
                self.trade_records.append(trade_record)
                
                # Record in agent's history if it's a BaseAgent
                if isinstance(self.agents[agent_name], BaseAgent):
                    self.agents[agent_name].record_trade(trade_record)
                    
                return True
                
        return False
        
    def get_leaderboard(self, current_price: float) -> List[Dict[str, Any]]:
        """Get performance leaderboard sorted by ROI."""
        results = []
        
        for agent_name in self.agents.keys():
            portfolio = self.portfolios[agent_name]
            initial_value = self.initial_values[agent_name]
            current_value = portfolio.get_total_value(current_price)
            roi = portfolio.get_roi(initial_value, current_price)
            
            results.append({
                'name': agent_name,
                'roi': roi,
                'current_value': current_value,
                'initial_value': initial_value,
                'initial_stock': self.initial_stocks.get(agent_name, 0),
                'cash': portfolio.cash,
                'stock': portfolio.stock,
                'trades': len([t for t in self.trade_records if t.agent_name == agent_name])
            })
            
        # Sort by ROI descending
        results.sort(key=lambda x: x['roi'], reverse=True)
        return results
        
    def get_portfolio_summary(self, current_price: float) -> Dict[str, Any]:
        """Get summary of all portfolios."""
        total_trades = len(self.trade_records)
        total_volume = sum(t.quantity for t in self.trade_records)
        
        return {
            'total_agents': len(self.agents),
            'total_trades': total_trades,
            'total_volume': total_volume,
            'leaderboard': self.get_leaderboard(current_price)
        }


# Example agents for testing
def create_sample_agents() -> List[TradingAgent]:
    """Create a set of sample agents for testing."""
    return [
        RandomAgent("RandomTrader1", aggressiveness=0.2),
        RandomAgent("RandomTrader2", aggressiveness=0.4),
        MomentumAgent("MomentumFollower", lookback_period=5),
        MeanReversionAgent("MeanReverter", window_size=8, threshold=0.015),
        MarketMakerAgent("MarketMaker1", spread_bps=8.0, quantity=2),
        MarketMakerAgent("MarketMaker2", spread_bps=12.0, quantity=1)
    ]


if __name__ == "__main__":
    # Test the agent management system
    print("🧪 Testing Agent Management System")
    
    # Create manager and add agents
    manager = AgentManager()
    sample_agents = create_sample_agents()
    
    for agent in sample_agents:
        manager.add_agent(agent)
        
    print(f"✅ Added {len(sample_agents)} agents")
    
    # Simulate a few ticks
    test_price = 150.0
    for tick in range(5):
        print(f"\\nTick {tick + 1}: Price = ${test_price:.2f}")
        
        # Get decisions
        orders = manager.get_agent_decisions(test_price, tick)
        print(f"📋 Generated {len(orders)} orders")
        
        # Execute some trades (simplified)
        for order in orders[:3]:  # Execute first 3 orders
            success = manager.execute_trade(
                order['agent'], order['side'], order['quantity'], test_price
            )
            if success:
                print(f"✅ Executed {order['side']} {order['quantity']} @ ${test_price:.2f} for {order['agent']}")
                
        # Update price slightly
        test_price += (tick - 2) * 0.5
        
    # Show final results
    print("\\n" + "="*50)
    print("📊 FINAL RESULTS")
    print("="*50)
    
    leaderboard = manager.get_leaderboard(test_price)
    for i, result in enumerate(leaderboard, 1):
        print(f"{i}. {result['name']}: ROI={result['roi']*100:+.2f}% | "
              f"Value=${result['current_value']:.2f} | Trades={result['trades']}")
