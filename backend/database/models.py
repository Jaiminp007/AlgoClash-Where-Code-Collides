"""
MongoDB Document Models
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import hashlib


@dataclass
class Generation:
    """Generation document"""
    generation_id: str
    selected_models: List[str] = field(default_factory=list)
    selected_stock: str = ""
    status: str = "pending"
    progress: int = 0
    message: str = ""
    algorithms: Dict[str, str] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)
    model_states: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Simulation:
    """Simulation document"""
    simulation_id: str
    generation_id: str
    stock_ticker: str = ""
    selected_models: List[str] = field(default_factory=list)
    status: str = "pending"
    progress: int = 0
    message: str = ""
    results: Optional[Dict] = None
    leaderboard: List[Dict] = field(default_factory=list)
    winner: Optional[Dict] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Algorithm:
    """Algorithm document - stores generated algorithms with their simulation performance"""
    simulation_id: str
    generation_id: str
    model_name: str
    code: str
    code_hash: str
    validation_status: str = "valid"
    validation_errors: Optional[List[str]] = None
    # Performance metrics
    performance_roi: Optional[float] = None
    performance_trades: int = 0
    initial_cash: float = 10000.0
    final_cash: Optional[float] = None
    initial_stock: int = 0
    final_stock: Optional[int] = None
    final_value: Optional[float] = None
    # Ranking info
    rank: Optional[int] = None
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    simulation_completed_at: Optional[datetime] = None

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def compute_hash(code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()


@dataclass
class SimulationTick:
    """Tick data"""
    simulation_id: str
    tick_number: int
    price: float
    timestamp: str
    agent_portfolios: Dict = field(default_factory=dict)
    trades: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self):
        return asdict(self)
