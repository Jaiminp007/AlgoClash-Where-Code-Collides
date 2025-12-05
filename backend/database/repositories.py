"""
MongoDB Repository Layer
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from .connection import get_db
from .models import Generation, Simulation, Algorithm, SimulationTick


class GenerationRepository:
    """Repository for generation documents"""

    def __init__(self):
        self.db = get_db()
        self.collection = self.db.generations

    def create(self, generation: Generation) -> str:
        """Create generation"""
        result = self.collection.insert_one(generation.to_dict())
        return str(result.inserted_id)

    def find_by_id(self, generation_id: str) -> Optional[Dict]:
        """Find generation by ID.

        Note: Model names have dots escaped as _DOT_ in storage.
        We unescape them when returning data.
        """
        result = self.collection.find_one({'generation_id': generation_id}, {'_id': 0})

        # Unescape dots in algorithm keys
        if result and 'algorithms' in result:
            unescaped_algorithms = {}
            for key, value in result['algorithms'].items():
                # Unescape _DOT_ back to .
                original_key = key.replace('_DOT_', '.')
                unescaped_algorithms[original_key] = value
            result['algorithms'] = unescaped_algorithms

        # Unescape dots in failures keys
        if result and 'failures' in result:
            unescaped_failures = {}
            for key, value in result['failures'].items():
                # Unescape _DOT_ back to .
                original_key = key.replace('_DOT_', '.')
                unescaped_failures[original_key] = value
            result['failures'] = unescaped_failures

        # Unescape dots in model_states keys
        if result and 'model_states' in result:
            unescaped_states = {}
            for key, value in result['model_states'].items():
                # Unescape _DOT_ back to .
                original_key = key.replace('_DOT_', '.')
                unescaped_states[original_key] = value
            result['model_states'] = unescaped_states

        return result

    def update_status(self, generation_id: str, status: str, progress: int = None, message: str = None):
        """Update generation status"""
        update_data = {
            'status': status,
            'updated_at': datetime.utcnow()
        }
        if progress is not None:
            update_data['progress'] = progress
        if message is not None:
            update_data['message'] = message

        self.collection.update_one(
            {'generation_id': generation_id},
            {'$set': update_data}
        )

    def add_algorithm(self, generation_id: str, model_name: str, code: str):
        """Add algorithm to generation.

        Note: MongoDB treats dots as nested path separators, so we escape them.
        The frontend must unescape them when displaying.
        """
        # Escape dots in model name to avoid MongoDB nested path issues
        # e.g., "anthropic/claude-haiku-4.5" -> "anthropic/claude-haiku-4_DOT_5"
        safe_model_name = model_name.replace('.', '_DOT_')

        self.collection.update_one(
            {'generation_id': generation_id},
            {
                '$set': {
                    f'algorithms.{safe_model_name}': code,
                    'updated_at': datetime.utcnow()
                }
            }
        )

    def add_failure(self, generation_id: str, model_name: str, reason: str):
        """Add failure reason for a model to generation."""
        safe_model_name = model_name.replace('.', '_DOT_')
        self.collection.update_one(
            {'generation_id': generation_id},
            {
                '$set': {
                    f'failures.{safe_model_name}': reason,
                    'updated_at': datetime.utcnow()
                }
            }
        )

    def set_error(self, generation_id: str, error: str):
        """Set error"""
        self.collection.update_one(
            {'generation_id': generation_id},
            {
                '$set': {
                    'status': 'error',
                    'error': error,
                    'updated_at': datetime.utcnow()
                }
            }
        )

    def update_model_state(self, generation_id: str, model_name: str, state: str):
        """Update state for a specific model"""
        safe_model_name = model_name.replace('.', '_DOT_')
        self.collection.update_one(
            {'generation_id': generation_id},
            {
                '$set': {
                    f'model_states.{safe_model_name}': state,
                    'updated_at': datetime.utcnow()
                }
            }
        )


class SimulationRepository:
    """Repository for simulation documents"""

    def __init__(self):
        self.db = get_db()
        self.collection = self.db.simulations

    def create(self, simulation: Simulation) -> str:
        """Create simulation"""
        result = self.collection.insert_one(simulation.to_dict())
        return str(result.inserted_id)

    def find_by_id(self, simulation_id: str) -> Optional[Dict]:
        """Find simulation by ID"""
        return self.collection.find_one({'simulation_id': simulation_id}, {'_id': 0})

    def update_status(self, simulation_id: str, status: str, progress: int = None, message: str = None):
        """Update simulation status"""
        update_data = {
            'status': status,
            'updated_at': datetime.utcnow()
        }
        if progress is not None:
            update_data['progress'] = progress
        if message is not None:
            update_data['message'] = message

        self.collection.update_one(
            {'simulation_id': simulation_id},
            {'$set': update_data}
        )

    def save_results(self, simulation_id: str, results: Dict):
        """Save simulation results"""
        # Extract winner from leaderboard
        winner = None
        if results.get('leaderboard') and len(results['leaderboard']) > 0:
            winner = results['leaderboard'][0]

        self.collection.update_one(
            {'simulation_id': simulation_id},
            {
                '$set': {
                    'status': 'completed',
                    'results': results,
                    'leaderboard': results.get('leaderboard', []),
                    'winner': winner,
                    'completed_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                }
            }
        )

    def set_error(self, simulation_id: str, error: str):
        """Set error"""
        self.collection.update_one(
            {'simulation_id': simulation_id},
            {
                '$set': {
                    'status': 'error',
                    'error': error,
                    'updated_at': datetime.utcnow()
                }
            }
        )

    def get_recent(self, limit: int = 10) -> List[Dict]:
        """Get recent simulations"""
        return list(self.collection.find(
            {},
            {'_id': 0}
        ).sort('created_at', -1).limit(limit))


class AlgorithmRepository:
    """Repository for algorithm documents"""

    def __init__(self):
        self.db = get_db()
        self.collection = self.db.algorithms

    def create(self, algorithm: Algorithm) -> str:
        """Create algorithm"""
        result = self.collection.insert_one(algorithm.to_dict())
        return str(result.inserted_id)

    def find_by_simulation(self, simulation_id: str) -> List[Dict]:
        """Find algorithms by simulation ID"""
        return list(self.collection.find(
            {'simulation_id': simulation_id},
            {'_id': 0}
        ))

    def find_by_generation(self, generation_id: str) -> List[Dict]:
        """Find algorithms by generation ID"""
        return list(self.collection.find(
            {'generation_id': generation_id},
            {'_id': 0}
        ))

    def update_performance(self, simulation_id: str, model_name: str, roi: float, trades: int):
        """Update algorithm performance"""
        self.collection.update_one(
            {'simulation_id': simulation_id, 'model_name': model_name},
            {
                '$set': {
                    'performance_roi': roi,
                    'performance_trades': trades
                }
            }
        )

    def save_from_simulation_results(
        self,
        simulation_id: str,
        generation_id: str,
        leaderboard: List[Dict],
        algorithms_code: Dict[str, str]
    ) -> int:
        """
        Save all algorithms from a simulation with their performance metrics.
        
        Args:
            simulation_id: The simulation ID
            generation_id: The generation ID
            leaderboard: List of dicts with name, roi, cash, stock, trades, etc.
            algorithms_code: Dict mapping agent names to their code
            
        Returns:
            Number of algorithms saved
        """
        saved_count = 0
        completed_at = datetime.utcnow()
        
        for rank, entry in enumerate(leaderboard, 1):
            agent_name = entry.get('name', '')
            
            # Try to find the code for this agent
            code = algorithms_code.get(agent_name, '')
            if not code:
                # Try without the prefix
                clean_name = agent_name.replace('generated_algo_', '')
                code = algorithms_code.get(clean_name, '')
            
            if not code:
                print(f"⚠️ No code found for agent: {agent_name}")
                continue
            
            algo = Algorithm(
                simulation_id=simulation_id,
                generation_id=generation_id,
                model_name=agent_name,
                code=code,
                code_hash=Algorithm.compute_hash(code),
                performance_roi=entry.get('roi'),
                performance_trades=entry.get('trades', 0),
                initial_cash=entry.get('initial_value', 10000.0),
                final_cash=entry.get('cash'),
                initial_stock=entry.get('initial_stock', 0),
                final_stock=entry.get('stock'),
                final_value=entry.get('current_value'),
                rank=rank,
                simulation_completed_at=completed_at
            )
            
            try:
                self.create(algo)
                saved_count += 1
                print(f"  💾 Saved algorithm: {agent_name} (rank #{rank}, ROI: {entry.get('roi', 0)*100:+.2f}%)")
            except Exception as e:
                print(f"  ❌ Failed to save {agent_name}: {e}")
        
        return saved_count

    def get_top_algorithms(self, limit: int = 10) -> List[Dict]:
        """Get top performing algorithms by ROI"""
        return list(self.collection.find(
            {'performance_roi': {'$ne': None}},
            {'_id': 0}
        ).sort('performance_roi', -1).limit(limit))

    def get_algorithm_history(self, model_name: str, limit: int = 20) -> List[Dict]:
        """Get algorithm history for a specific model"""
        return list(self.collection.find(
            {'model_name': {'$regex': model_name, '$options': 'i'}},
            {'_id': 0}
        ).sort('created_at', -1).limit(limit))


class TickRepository:
    """Repository for simulation ticks"""

    def __init__(self):
        self.db = get_db()
        self.collection = self.db.simulation_ticks

    def save_tick(self, tick: SimulationTick):
        """Save tick data"""
        self.collection.insert_one(tick.to_dict())

    def get_ticks(self, simulation_id: str) -> List[Dict]:
        """Get all ticks for simulation"""
        return list(self.collection.find(
            {'simulation_id': simulation_id},
            {'_id': 0}
        ).sort('tick_number', 1))

    def delete_by_simulation(self, simulation_id: str):
        """Delete ticks for simulation"""
        self.collection.delete_many({'simulation_id': simulation_id})
