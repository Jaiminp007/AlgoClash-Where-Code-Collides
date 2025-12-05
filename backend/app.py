"""
Flask Application with MongoDB Integration
This file shows the MongoDB-integrated version of app.py

To use this version:
1. Install dependencies: pip install pymongo python-dotenv
2. Ensure MongoDB is running at mongodb://localhost:27017/ai_trader_battlefield
3. Replace app.py with this file or merge the changes
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import shutil
import uuid

# MongoDB imports
from database import (
    init_db,
    get_db,
    Generation,
    Simulation,
    Algorithm,
    SimulationTick,
    GenerationRepository,
    SimulationRepository,
    AlgorithmRepository,
    TickRepository
)

# Ensure backend/.env is loaded when running the Flask app
try:
    from dotenv import load_dotenv
    load_dotenv()  # CWD
    _env_path = Path(__file__).resolve().parent / '.env'
    if _env_path.exists():
        load_dotenv(_env_path)
except Exception:
    pass

app = Flask(__name__)
CORS(app)

GENERATION_STALE_TIMEOUT_SECONDS = int(os.getenv("GENERATION_STALE_TIMEOUT_SECONDS", "180"))

# Initialize MongoDB on app startup
@app.before_request
def initialize_database():
    """Initialize MongoDB connection before first request"""
    try:
        init_db()
        print("✅ MongoDB initialized successfully")
    except Exception as e:
        print(f"❌ MongoDB initialization failed: {e}")
        raise

# Initialize repositories
gen_repo = GenerationRepository()
sim_repo = SimulationRepository()
algo_repo = AlgorithmRepository()
tick_repo = TickRepository()


def _mark_pending_models_as_failed(
    gen_id: str,
    generation: dict | None = None,
    reason: str = "Generation timed out before model completed",
    models: list[str] | None = None,
) -> list[str]:
    """Ensure every model ends in either algorithms or failures even if the worker crashed."""
    generation = generation or gen_repo.find_by_id(gen_id)
    if not generation:
        return []

    algorithms = generation.get("algorithms", {})
    failures = generation.get("failures", {})
    model_states = generation.get("model_states", {})
    pending: list[str] = []

    target_models = models or generation.get("selected_models", [])

    for model in target_models:
        if model in algorithms or model in failures:
            continue
        if model_states.get(model) == "error":
            continue

        pending.append(model)
        gen_repo.add_failure(gen_id, model, reason)
        gen_repo.update_model_state(gen_id, model, "error")

    return pending


def _handle_stale_generation(gen_id: str, generation: dict) -> dict:
    """If a generation has not updated recently, mark unfinished models as timed-out failures."""
    if not generation:
        return generation

    if generation.get("status") != "generating":
        return generation

    updated_at = generation.get("updated_at")
    if not updated_at:
        return generation

    if isinstance(updated_at, str):
        try:
            updated_at = datetime.fromisoformat(updated_at)
        except ValueError:
            return generation

    if datetime.utcnow() - updated_at < timedelta(seconds=GENERATION_STALE_TIMEOUT_SECONDS):
        return generation

    pending = _mark_pending_models_as_failed(
        gen_id,
        generation=generation,
        reason="Generation timed out while waiting on the model response"
    )

    if pending:
        # Check if we have at least some successful algorithms
        algorithms = generation.get("algorithms", {})
        if len(algorithms) >= 2:
            # We have enough algorithms to proceed - mark as completed with warning
            gen_repo.update_status(
                gen_id,
                "completed",
                100,
                f"Generation completed ({len(algorithms)} successful, {len(pending)} timed out)"
            )
        else:
            # Not enough algorithms - mark as error
            gen_repo.update_status(
                gen_id,
                "error",
                generation.get("progress", 0),
                f"Timed out waiting on {len(pending)} model(s): {', '.join(pending)}"
            )
        return gen_repo.find_by_id(gen_id) or generation

    return generation

# Register algorithm preview/management blueprint
from api.algos import algos_bp
app.register_blueprint(algos_bp)

# Health check
@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "database": "mongodb"})

# Serve AI agents JSON from backend/open_router/ai_agents.json
@app.get("/api/ai_agents")
def get_ai_agents():
    json_path = Path(__file__).resolve().parent / "open_router" / "ai_agents.json"
    if not json_path.exists():
        return jsonify({"error": f"ai_agents.json not found at {str(json_path)}"}), 404
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/api/data_files")
def list_data_files():
    """List available stocks from stock_ticker.json in open_router folder"""
    try:
        json_path = Path(__file__).resolve().parent / "open_router" / "stock_ticker.json"
        if not json_path.exists():
            return jsonify({"error": "stock_ticker.json not found"}), 404

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tickers = data.get("Stock_Ticker", [])
        stocks = [{
            "ticker": ticker.upper(),
            "filename": ticker.upper()  # Kept as filename for frontend compatibility, but it's just the ticker
        } for ticker in tickers]

        return jsonify({"stocks": stocks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/api/generate")
def generate_algorithms():
    """Generate algorithms only (no simulation) - MongoDB version"""
    try:
        data = request.get_json()
        agents = data.get('agents', [])
        stock = data.get('stock', 'AAPL')  # Default to AAPL ticker

        if len(agents) < 2:
            return jsonify({"error": "At least 2 agents are required"}), 400

        # Generate unique generation ID
        gen_id = str(uuid.uuid4())

        # Create Generation document
        generation = Generation(
            generation_id=gen_id,
            selected_models=agents,
            selected_stock=stock,
            status="starting",
            progress=0,
            message="Initializing algorithm generation..."
        )

        # Save to MongoDB
        gen_repo.create(generation)

        # Start generation in background thread
        thread = threading.Thread(
            target=run_generation_background,
            args=(gen_id, agents, stock)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            "generation_id": gen_id,
            "status": "started",
            "message": "Algorithm generation started"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/api/generation/<gen_id>")
def get_generation_status(gen_id):
    """Get status of algorithm generation - MongoDB version"""
    generation = gen_repo.find_by_id(gen_id)

    if not generation:
        return jsonify({"error": "Generation not found"}), 404

    generation = _handle_stale_generation(gen_id, generation)

    return jsonify(generation)

@app.route("/api/generation/<gen_id>/regenerate", methods=['POST'])
def regenerate_single_algorithm(gen_id):
    """Regenerate a single algorithm for a specific model - MongoDB version"""
    try:
        print(f"\n{'='*60}")
        print(f"📥 REGENERATE REQUEST for generation: {gen_id}")

        generation = gen_repo.find_by_id(gen_id)
        if not generation:
            print(f"❌ Generation {gen_id} not found")
            return jsonify({"error": "Generation not found"}), 404

        data = request.get_json()
        old_model = data.get('old_model')
        new_model = data.get('new_model')

        print(f"🔄 Replacing: {old_model} → {new_model}")

        if not old_model or not new_model:
            return jsonify({"error": "Both old_model and new_model are required"}), 400

        # Update the agent list in MongoDB
        agents = generation.get("selected_models", [])
        
        # Try to find old_model in agents list
        # Handle potential _DOT_ mismatch if frontend sent escaped name
        target_idx = -1
        if old_model in agents:
            target_idx = agents.index(old_model)
        elif old_model.replace('_DOT_', '.') in agents:
            target_idx = agents.index(old_model.replace('_DOT_', '.'))
            
        if target_idx >= 0:
            agents[target_idx] = new_model

            # Update in database
            get_db().generations.update_one(
                {'generation_id': gen_id},
                {'$set': {'selected_models': agents}}
            )
            print(f"📝 Updated agents list: replaced position {target_idx}")
        else:
            print(f"⚠️ Warning: Could not find {old_model} in agents list {agents}")

        # Remove old algorithm from generation.algorithms
        # Handle both escaped and unescaped versions
        safe_old_model = old_model.replace('.', '_DOT_')
        get_db().generations.update_one(
            {'generation_id': gen_id},
            {'$unset': {
                f'algorithms.{old_model}': "",
                f'algorithms.{safe_old_model}': ""
            }}
        )

        # Remove old failure from generation.failures if it exists
        # This ensures the frontend doesn't show "Failed" while we are retrying
        get_db().generations.update_one(
            {'generation_id': gen_id},
            {'$unset': {
                f'failures.{old_model}': "",
                f'failures.{safe_old_model}': ""
            }}
        )

        # Update model_states: remove old model's state and set new model to 'generating'
        get_db().generations.update_one(
            {'generation_id': gen_id},
            {
                '$unset': {
                    f'model_states.{old_model}': "",
                    f'model_states.{safe_old_model}': ""
                },
                '$set': {
                    f'model_states.{new_model}': 'generating',
                    # Mark overall generation as running again so frontend knows work is in progress
                    'status': 'running',
                    'progress': max(generation.get('progress', 40), 40),
                    'message': f'Regenerating algorithm for {new_model}...'
                }
            }
        )
        print(f"📝 Updated model_states: removed {old_model}, set {new_model} to 'generating'")

        # Start regeneration in background thread
        thread = threading.Thread(
            target=regenerate_single_algorithm_background,
            args=(gen_id, new_model, generation.get("selected_stock", "AAPL"))
        )
        thread.daemon = True
        thread.start()

        print(f"✅ Started regeneration thread")
        print(f"{'='*60}\n")

        return jsonify({
            "status": "started",
            "message": f"Regenerating algorithm for {new_model}"
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.post("/api/simulate/<gen_id>")
def simulate_with_algorithms(gen_id):
    """Run simulation with pre-generated algorithms - MongoDB version
    
    Adaptive mode is enabled by default with checkpoints at ticks 130 and 260.
    At each checkpoint, agents with negative ROI can improve their algorithms.
    """
    try:
        generation = gen_repo.find_by_id(gen_id)

        if not generation:
            return jsonify({"error": "Generation not found"}), 404

        # Allow simulation to run as long as we have at least one
        # generated algorithm, even if the overall generation status
        # is not strictly "completed" (e.g. some models failed).
        algorithms = generation.get("algorithms", {}) or {}
        if not algorithms:
            return jsonify({"error": "No algorithms available to simulate"}), 400

        # Generate unique simulation ID
        sim_id = str(uuid.uuid4())

        # Extract ticker from filename (or use directly if it is a ticker)
        stock_val = generation["selected_stock"]
        ticker = stock_val.upper()

        # Create Simulation document
        simulation = Simulation(
            simulation_id=sim_id,
            generation_id=gen_id,
            stock_ticker=ticker,
            selected_models=generation.get("selected_models", []),
            status="starting",
            progress=0,
            message="Initializing simulation with adaptive checkpoints..."
        )

        # Save to MongoDB
        sim_repo.create(simulation)

        # Start simulation in background thread (adaptive mode is now standard)
        thread = threading.Thread(
            target=run_simulation_only_background,
            args=(sim_id, gen_id, ticker, generation.get("selected_models", []))
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            "simulation_id": sim_id,
            "status": "started",
            "message": "Simulation started with adaptive checkpoints at ticks 130 and 260",
            "adaptive_mode": True,
            "checkpoints": [130, 260]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/api/simulation/<sim_id>")
def get_simulation_status(sim_id):
    """Get status of a running simulation - MongoDB version"""
    simulation = sim_repo.find_by_id(sim_id)

    if not simulation:
        return jsonify({"error": "Simulation not found"}), 404

    # Fetch recent ticks to build chart data
    # Only fetch if simulation is running or completed
    if simulation.get("status") in ["running", "completed"]:
        ticks = tick_repo.get_ticks(sim_id)
        
        # Format ticks for frontend chart
        chart_data = []
        for tick in ticks:
            chart_point = {
                "tick": tick.get("tick_number"),
                "price": tick.get("price"),
                "timestamp": tick.get("timestamp"),
                "agent_portfolios": tick.get("agent_portfolios"),
                "trades": tick.get("trades")
            }
            chart_data.append(chart_point)
            
        simulation["chart_data"] = chart_data

    return jsonify(simulation)


@app.post("/api/cleanup/<gen_id>")
def cleanup_generated_files(gen_id):
    """
    Cleanup generated algorithm files from the filesystem.
    Called when user clicks 'Back to Dashboard' after viewing results.
    Algorithms are already saved in MongoDB, so we just need to remove the files.
    """
    try:
        gen_dir = Path(__file__).resolve().parent / "generate_algo"
        
        if gen_dir.exists() and gen_dir.is_dir():
            shutil.rmtree(gen_dir, ignore_errors=True)
            print(f"🧹 Cleaned up generated algorithms folder for generation {gen_id}")
            return jsonify({
                "success": True,
                "message": "Generated algorithm files cleaned up successfully",
                "generation_id": gen_id
            })
        else:
            return jsonify({
                "success": True,
                "message": "No files to clean up",
                "generation_id": gen_id
            })
            
    except Exception as e:
        print(f"⚠️ Cleanup error for generation {gen_id}: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "generation_id": gen_id
        }), 500


@app.get("/api/algorithms/<sim_id>")
def get_simulation_algorithms(sim_id):
    """Get all algorithms stored for a simulation"""
    try:
        algorithms = algo_repo.find_by_simulation(sim_id)
        return jsonify({
            "simulation_id": sim_id,
            "algorithms": algorithms,
            "count": len(algorithms)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/algorithms/top")
def get_top_algorithms():
    """Get top performing algorithms across all simulations"""
    try:
        limit = request.args.get('limit', 10, type=int)
        algorithms = algo_repo.get_top_algorithms(limit)
        return jsonify({
            "algorithms": algorithms,
            "count": len(algorithms)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Background task functions

def run_generation_background(gen_id, agents, stock_val):
    """Generate algorithms in background thread - MongoDB version"""
    try:
        print(f"🔄 Starting generation for {len(agents)} agents in generation {gen_id}")

        # Update status
        gen_repo.update_status(gen_id, "generating", 10, "Generating algorithms...")

        # Import generation function
        from open_router.algo_gen import generate_algorithms_for_agents

        # Extract ticker from filename (or use directly if it is a ticker)
        ticker = stock_val.upper()
        print(f"📊 Using ticker: {ticker}")

        # Progress callback
        total_agents = len(agents)
        completed_count = [0]  # Using list for mutable counter in closure

        def progress_callback(progress, message):
            try:
                # Handle JSON-encoded preview (new format)
                if isinstance(message, str) and message.startswith("PREVIEW_JSON::"):
                    json_data = message.replace("PREVIEW_JSON::", "", 1)
                    data = json.loads(json_data)
                    model = data["model"]
                    code = data["code"]
                    # Save algorithm to generation document
                    gen_repo.add_algorithm(gen_id, model, code)
                    gen_repo.update_model_state(gen_id, model, "done")

                    # Also save as separate Algorithm document
                    algo = Algorithm(
                        simulation_id="",  # Not in simulation yet
                        generation_id=gen_id,
                        model_name=model,
                        code=code,
                        code_hash=Algorithm.compute_hash(code),
                        validation_status="valid"
                    )
                    algo_repo.create(algo)

                    completed_count[0] += 1
                    overall_progress = 10 + int((completed_count[0] / total_agents) * 80)
                    gen_repo.update_status(gen_id, "generating", overall_progress,
                                          f"Generated {completed_count[0]}/{total_agents} algorithms")

                # Handle old string format for backwards compatibility
                elif isinstance(message, str) and message.startswith("PREVIEW::"):
                    parts = message.split("::", 2)
                    if len(parts) == 3:
                        _, model, code = parts
                        # Save algorithm to generation document
                        gen_repo.add_algorithm(gen_id, model, code)
                        gen_repo.update_model_state(gen_id, model, "done")

                        # Also save as separate Algorithm document
                        algo = Algorithm(
                            simulation_id="",  # Not in simulation yet
                            generation_id=gen_id,
                            model_name=model,
                            code=code,
                            code_hash=Algorithm.compute_hash(code),
                            validation_status="valid"
                        )
                        algo_repo.create(algo)

                        completed_count[0] += 1
                        overall_progress = 10 + int((completed_count[0] / total_agents) * 80)
                        gen_repo.update_status(gen_id, "generating", overall_progress,
                                              f"Generated {completed_count[0]}/{total_agents} algorithms")

                # Handle failure messages
                elif isinstance(message, str) and message.startswith("MODEL_FAIL::"):
                    parts = message.split("::", 2)
                    if len(parts) >= 2:
                        model = parts[1]
                        reason = parts[2] if len(parts) > 2 else "Unknown error"
                        gen_repo.add_failure(gen_id, model, reason)
                        gen_repo.update_model_state(gen_id, model, "error")
                        print(f"❌ Recorded failure for {model}: {reason}")

                # Handle start messages
                elif isinstance(message, str) and message.startswith("MODEL_START::"):
                    parts = message.split("::", 1)
                    if len(parts) >= 2:
                        model = parts[1]
                        gen_repo.update_model_state(gen_id, model, "generating")

                # Handle skip messages
                elif isinstance(message, str) and message.startswith("MODEL_SKIP::"):
                    parts = message.split("::", 2)
                    if len(parts) >= 2:
                        model = parts[1]
                        reason = parts[2] if len(parts) > 2 else "Unavailable"
                        gen_repo.add_failure(gen_id, model, reason)
                        gen_repo.update_model_state(gen_id, model, "error")

                # Handle generic progress messages
                elif isinstance(message, str):
                    # Update the main status message so frontend can parse it if needed
                    gen_repo.update_status(gen_id, "generating", progress, message)

            except Exception as e:
                print(f"Progress callback error: {e}")

        # Generate algorithms
        success = generate_algorithms_for_agents(agents, ticker, progress_callback)
        pending = _mark_pending_models_as_failed(
            gen_id,
            reason="Generation worker stopped before the model finished",
            models=agents
        )

        if success and not pending:
            gen_repo.update_status(gen_id, "completed", 100, "All algorithms generated successfully")
        else:
            if pending:
                message = (
                    f"Generation finished with failures for {len(pending)} model(s): "
                    + ", ".join(pending)
                )
            else:
                message = "Some algorithms failed to generate"

            final_status = "completed" if success else "error"
            gen_repo.update_status(gen_id, final_status, 100, message)

    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        gen_repo.set_error(gen_id, str(e))


def regenerate_single_algorithm_background(gen_id, model, stock_val):
    """Regenerate a single algorithm in background thread - MongoDB version"""
    try:
        print(f"🔄 Starting regeneration for {model} in generation {gen_id}")

        # Import generation function
        from open_router.algo_gen import generate_algorithms_for_agents

        # Extract ticker from filename (or use directly if it is a ticker)
        ticker = stock_val.upper()
        print(f"📊 Using ticker: {ticker}")

        # Progress callback for single model
        def progress_callback(progress, message):
            try:
                # Handle JSON-encoded preview (new format)
                if isinstance(message, str) and message.startswith("PREVIEW_JSON::"):
                    json_data = message.replace("PREVIEW_JSON::", "", 1)
                    data = json.loads(json_data)
                    msg_model = data["model"]
                    code = data["code"]
                    if msg_model == model:
                        # Update generation document
                        gen_repo.add_algorithm(gen_id, model, code)
                        gen_repo.update_model_state(gen_id, model, "done")

                        # Create/update Algorithm document
                        algo = Algorithm(
                            simulation_id="",
                            generation_id=gen_id,
                            model_name=model,
                            code=code,
                            code_hash=Algorithm.compute_hash(code),
                            validation_status="valid"
                        )
                        algo_repo.create(algo)

                # Handle old string format for backwards compatibility
                elif isinstance(message, str) and message.startswith("PREVIEW::"):
                    parts = message.split("::", 2)
                    if len(parts) == 3:
                        _, msg_model, code = parts
                        if msg_model == model:
                            # Update generation document
                            gen_repo.add_algorithm(gen_id, model, code)
                            gen_repo.update_model_state(gen_id, model, "done")

                            # Create/update Algorithm document
                            algo = Algorithm(
                                simulation_id="",
                                generation_id=gen_id,
                                model_name=model,
                                code=code,
                                code_hash=Algorithm.compute_hash(code),
                            validation_status="valid"
                        )
                        algo_repo.create(algo)

                # Handle failure messages
                elif isinstance(message, str) and message.startswith("MODEL_FAIL::"):
                    parts = message.split("::", 2)
                    if len(parts) >= 2:
                        msg_model = parts[1]
                        reason = parts[2] if len(parts) > 2 else "Unknown error"
                        if msg_model == model:
                            gen_repo.add_failure(gen_id, model, reason)
                            gen_repo.update_model_state(gen_id, model, "error")
                            print(f"❌ Recorded failure for {model}: {reason}")

                # Handle start messages
                elif isinstance(message, str) and message.startswith("MODEL_START::"):
                    parts = message.split("::", 1)
                    if len(parts) >= 2:
                        msg_model = parts[1]
                        if msg_model == model:
                            gen_repo.update_model_state(gen_id, model, "generating")

                # Handle skip messages
                elif isinstance(message, str) and message.startswith("MODEL_SKIP::"):
                    parts = message.split("::", 2)
                    if len(parts) >= 2:
                        msg_model = parts[1]
                        reason = parts[2] if len(parts) > 2 else "Unavailable"
                        if msg_model == model:
                            gen_repo.add_failure(gen_id, model, reason)
                            gen_repo.update_model_state(gen_id, model, "error")

            except Exception as e:
                print(f"Progress callback error: {e}")
        # Generate algorithm for single model
        print(f"🤖 Calling generate_algorithms_for_agents for {model}...")
        success = generate_algorithms_for_agents([model], ticker, progress_callback)
        print(f"{'✅' if success else '❌'} Generation result for {model}: {success}")

        pending = _mark_pending_models_as_failed(
            gen_id,
            reason="Regeneration worker stopped before the model finished",
            models=[model]
        )

        if pending:
            gen_repo.update_status(
                gen_id,
                "completed",
                100,
                f"Regeneration timed out for {', '.join(pending)}"
            )
        elif not success:
            gen_repo.update_status(gen_id, "completed", 100, f"Regeneration failed for {model}")

    except Exception as e:
        print(f"❌ Regeneration error: {e}")
        import traceback
        traceback.print_exc()
        gen_repo.update_status(gen_id, "completed", 100, f"Error regenerating {model}: {str(e)}")


def run_simulation_only_background(sim_id, gen_id, ticker, agents_list):
    """Run simulation with pre-generated algorithms - MongoDB version
    Now runs in ADAPTIVE mode by default with checkpoints at ticks 130 and 260.
    """
    try:
        print(f"\n{'='*60}")
        print(f"🚀 Starting simulation {sim_id} for {ticker}")
        print(f"🔄 ADAPTIVE MODE ENABLED (standard)")
        print(f"📊 Checkpoints at ticks: 130, 260")
        print(f"{'='*60}\n")

        # Update status
        sim_repo.update_status(sim_id, "running", 0, "Loading algorithms...")

        # Import simulation function
        from main import run_market_simulation
        from open_router.algo_gen import build_adaptation_prompt, regenerate_algorithm_for_adaptation_async
        import asyncio
        import httpx

        # Progress callback for simulation
        def progress_callback(progress, message):
            sim_repo.update_status(sim_id, "running", int(progress), message)

        # Tick callback to save tick data
        tick_data_buffer = []

        def tick_callback(tick_num, tick_data, trades):
            try:
                # Create tick document
                tick_doc = SimulationTick(
                    simulation_id=sim_id,
                    tick_number=tick_num,
                    price=tick_data.get('price', 0),
                    timestamp=tick_data.get('timestamp', ''),
                    agent_portfolios=tick_data.get('agent_portfolios', {}),
                    trades=[{
                        'buy_agent': t.buy_agent,
                        'sell_agent': t.sell_agent,
                        'quantity': t.quantity,
                        'price': t.price,
                        'timestamp': t.timestamp
                    } for t in trades] if trades else []
                )

                # Buffer ticks and save in batches
                tick_data_buffer.append(tick_doc)
                if len(tick_data_buffer) >= 10:  # Save every 10 ticks
                    for tick in tick_data_buffer:
                        tick_repo.save_tick(tick)
                    tick_data_buffer.clear()

            except Exception as e:
                print(f"Tick callback error: {e}")

        # Adaptation callback - called at checkpoints (130, 260)
        async def adaptation_callback_async(agents_data, checkpoint_num):
            """
            Called at each checkpoint (ticks 130, 260).
            Gives each AI agent a chance to analyze and improve their algorithm.
            """
            print(f"\n{'='*60}")
            print(f"🔄 ADAPTATION CHECKPOINT {checkpoint_num}/2")
            print(f"📊 Analyzing {len(agents_data)} agents...")
            print(f"{'='*60}")
            
            # Update simulation status
            sim_repo.update_status(
                sim_id, "running", 
                50 + checkpoint_num * 10,
                f"Checkpoint {checkpoint_num}: Agents analyzing performance..."
            )
            
            results = {}
            
            # Create HTTP client for API calls
            timeout = httpx.Timeout(45.0, connect=10.0)
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                for agent_data in agents_data:
                    agent_name = agent_data['agent_name']
                    roi_pct = agent_data['current_roi'] * 100
                    
                    # Build the adaptation prompt with trade history for detailed analysis
                    prompt = build_adaptation_prompt(
                        ticker=agent_data['ticker'],
                        current_algo_code=agent_data['current_code'],
                        current_roi=agent_data['current_roi'],
                        current_cash=agent_data['current_cash'],
                        current_shares=agent_data['current_shares'],
                        current_tick=agent_data['current_tick'],
                        total_ticks=agent_data['total_ticks'],
                        price_history=agent_data['price_history'],
                        checkpoint_num=checkpoint_num,
                        trades=agent_data.get('trades', []),
                        total_checkpoints=2  # Checkpoints at 130 and 260
                    )
                    
                    # Only adapt if ROI is negative (agent is losing)
                    if agent_data['current_roi'] < 0:  # Any loss triggers adaptation
                        print(f"  📤 {agent_name}: Requesting adaptation (ROI: {roi_pct:+.2f}%)")
                        
                        # Try to find the original model ID from the generation
                        generation = gen_repo.find_by_id(gen_id)
                        original_model_id = None
                        if generation:
                            for m in generation.get('selected_models', []):
                                sanitized = m.replace('/', '_').replace('-', '_').replace(':', '_').replace('.', '_')
                                if f"generated_algo_{sanitized}" == agent_name:
                                    original_model_id = m
                                    break
                        
                        if original_model_id:
                            should_update, new_code = await regenerate_algorithm_for_adaptation_async(
                                client, original_model_id, prompt
                            )
                            
                            if should_update and new_code:
                                results[agent_name] = new_code
                                print(f"  ✅ {agent_name}: Received improved algorithm")
                            else:
                                print(f"  ➡️ {agent_name}: Keeping current algorithm")
                        else:
                            print(f"  ⚠️ {agent_name}: Could not find original model ID")
                    else:
                        status = "✅" if roi_pct >= 0 else "⚠️"
                        print(f"  {status} {agent_name}: ROI={roi_pct:+.2f}% - no adaptation needed")
            
            print(f"{'='*60}\n")
            return results
        
        def adaptation_callback(agents_data, checkpoint_num):
            """Sync wrapper for async adaptation callback."""
            return adaptation_callback_async(agents_data, checkpoint_num)

        # Run simulation WITH ADAPTATION ENABLED (standard)
        results = run_market_simulation(
            ticker,
            progress_callback=progress_callback,
            tick_callback=tick_callback,
            allowed_models=agents_list,
            enable_adaptation=True,
            adaptation_callback=adaptation_callback
        )

        # Save any remaining ticks
        for tick in tick_data_buffer:
            tick_repo.save_tick(tick)

        # Add adaptation info to results
        if results and isinstance(results, dict):
            results['adaptation_enabled'] = True
            results['checkpoints'] = [130, 260]

        # Save results to database
        sim_repo.save_results(sim_id, results)

        # Store algorithms to MongoDB with their performance metrics
        if results and results.get('leaderboard'):
            print(f"\n💾 Saving algorithms to MongoDB...")
            generation = gen_repo.find_by_id(gen_id)
            if generation:
                algorithms_code = generation.get('algorithms', {})
                # Map agent names to their code
                algo_code_map = {}
                for model_name, code in algorithms_code.items():
                    # Create the sanitized agent name format
                    sanitized = model_name.replace('/', '_').replace('-', '_').replace(':', '_').replace('.', '_')
                    agent_name = f"generated_algo_{sanitized}"
                    algo_code_map[agent_name] = code
                
                saved_count = algo_repo.save_from_simulation_results(
                    simulation_id=sim_id,
                    generation_id=gen_id,
                    leaderboard=results['leaderboard'],
                    algorithms_code=algo_code_map
                )
                print(f"💾 Saved {saved_count} algorithms to MongoDB")

        print(f"✅ Simulation {sim_id} completed successfully")

    except Exception as e:
        print(f"❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()
        sim_repo.set_error(sim_id, str(e))


@app.post("/api/simulate/<gen_id>/adaptive")
def simulate_with_adaptation(gen_id):
    """Run simulation with pre-generated algorithms AND mid-simulation adaptation - MongoDB version"""
    try:
        generation = gen_repo.find_by_id(gen_id)

        if not generation:
            return jsonify({"error": "Generation not found"}), 404

        if generation["status"] != "completed":
            return jsonify({"error": "Algorithm generation not completed"}), 400

        # Generate unique simulation ID
        sim_id = str(uuid.uuid4())

        # Extract ticker from filename (or use directly if it is a ticker)
        stock_val = generation["selected_stock"]
        ticker = stock_val.upper()

        # Create Simulation document
        simulation = Simulation(
            simulation_id=sim_id,
            generation_id=gen_id,
            stock_ticker=ticker,
            selected_models=generation.get("selected_models", []),
            status="starting",
            progress=0,
            message="Initializing adaptive simulation..."
        )

        # Save to MongoDB
        sim_repo.create(simulation)

        # Start simulation in background thread WITH adaptation enabled
        thread = threading.Thread(
            target=run_adaptive_simulation_background,
            args=(sim_id, gen_id, ticker, generation.get("selected_models", []))
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            "simulation_id": sim_id,
            "status": "started",
            "message": "Adaptive simulation started (with algorithm improvement at checkpoints)",
            "checkpoints": [130, 260]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_adaptive_simulation_background(sim_id, gen_id, ticker, agents_list):
    """Run simulation with mid-simulation adaptation enabled - MongoDB version"""
    try:
        print(f"\n{'='*60}")
        print(f"🚀 Starting ADAPTIVE simulation {sim_id} for {ticker}")
        print(f"🔄 Adaptation checkpoints: [130, 260] ticks")
        print(f"📊 Agents will be analyzed at each checkpoint")
        print(f"{'='*60}\n")

        # Update status
        sim_repo.update_status(sim_id, "running", 0, "Loading algorithms (ADAPTIVE MODE)...")

        # Import simulation function
        from main import run_market_simulation
        from open_router.algo_gen import build_adaptation_prompt, regenerate_algorithm_for_adaptation_async
        import asyncio
        import httpx

        # Progress callback for simulation
        def progress_callback(progress, message):
            sim_repo.update_status(sim_id, "running", int(progress), message)

        # Tick callback to save tick data
        tick_data_buffer = []

        def tick_callback(tick_num, tick_data, trades):
            try:
                # Create tick document
                tick_doc = SimulationTick(
                    simulation_id=sim_id,
                    tick_number=tick_num,
                    price=tick_data.get('price', 0),
                    timestamp=tick_data.get('timestamp', ''),
                    agent_portfolios=tick_data.get('agent_portfolios', {}),
                    trades=[{
                        'buy_agent': t.buy_agent,
                        'sell_agent': t.sell_agent,
                        'quantity': t.quantity,
                        'price': t.price,
                        'timestamp': t.timestamp
                    } for t in trades] if trades else []
                )

                # Buffer ticks and save in batches
                tick_data_buffer.append(tick_doc)
                if len(tick_data_buffer) >= 10:
                    for tick in tick_data_buffer:
                        tick_repo.save_tick(tick)
                    tick_data_buffer.clear()

            except Exception as e:
                print(f"Tick callback error: {e}")

        # Adaptation callback - called at checkpoints
        async def adaptation_callback_async(agents_data, checkpoint_num):
            """
            Called at each checkpoint (ticks 130, 260).
            Gives each AI agent a chance to analyze and improve their algorithm.
            """
            print(f"\n🔄 ADAPTATION CHECKPOINT {checkpoint_num}")
            print(f"📊 Analyzing {len(agents_data)} agents...")
            
            # Update simulation status
            sim_repo.update_status(
                sim_id, "running", 
                50 + checkpoint_num * 10,
                f"Checkpoint {checkpoint_num}: Agents analyzing performance..."
            )
            
            results = {}
            
            # Create HTTP client for API calls
            timeout = httpx.Timeout(45.0, connect=10.0)
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                for agent_data in agents_data:
                    agent_name = agent_data['agent_name']
                    
                    # Extract model ID from agent name (remove 'generated_algo_' prefix and unescape)
                    # generated_algo_google_gemini_2_5_flash_lite -> google/gemini-2.5-flash-lite
                    model_id = agent_name.replace('generated_algo_', '')
                    model_id = model_id.replace('_', '/')  # First pass
                    # Handle special cases like google/gemini/2/5/flash/lite -> google/gemini-2.5-flash-lite
                    # This is approximate - exact reverse mapping would require storing original IDs
                    
                    # Build the adaptation prompt with trade history for detailed analysis
                    prompt = build_adaptation_prompt(
                        ticker=agent_data['ticker'],
                        current_algo_code=agent_data['current_code'],
                        current_roi=agent_data['current_roi'],
                        current_cash=agent_data['current_cash'],
                        current_shares=agent_data['current_shares'],
                        current_tick=agent_data['current_tick'],
                        total_ticks=agent_data['total_ticks'],
                        price_history=agent_data['price_history'],
                        checkpoint_num=checkpoint_num,
                        trades=agent_data.get('trades', []),
                        total_checkpoints=2  # Checkpoints at 130 and 260
                    )
                    
                    # Only adapt if ROI is negative (agent is losing)
                    if agent_data['current_roi'] < -0.02:  # More than 2% loss
                        print(f"  📤 {agent_name}: Requesting adaptation (ROI: {agent_data['current_roi']*100:+.2f}%)")
                        
                        # Try to find the original model ID from the generation
                        generation = gen_repo.find_by_id(gen_id)
                        original_model_id = None
                        if generation:
                            for m in generation.get('selected_models', []):
                                sanitized = m.replace('/', '_').replace('-', '_').replace(':', '_').replace('.', '_')
                                if f"generated_algo_{sanitized}" == agent_name:
                                    original_model_id = m
                                    break
                        
                        if original_model_id:
                            should_update, new_code = await regenerate_algorithm_for_adaptation_async(
                                client, original_model_id, prompt
                            )
                            
                            if should_update and new_code:
                                results[agent_name] = new_code
                                print(f"  ✅ {agent_name}: Received improved algorithm")
                            else:
                                print(f"  ➡️ {agent_name}: Keeping current algorithm")
                        else:
                            print(f"  ⚠️ {agent_name}: Could not find original model ID")
                    else:
                        print(f"  ✅ {agent_name}: Performing well (ROI: {agent_data['current_roi']*100:+.2f}%), no adaptation needed")
            
            return results
        
        def adaptation_callback(agents_data, checkpoint_num):
            """Sync wrapper for async adaptation callback."""
            return adaptation_callback_async(agents_data, checkpoint_num)

        # Run simulation with adaptation enabled
        results = run_market_simulation(
            ticker,
            progress_callback=progress_callback,
            tick_callback=tick_callback,
            allowed_models=agents_list,
            enable_adaptation=True,
            adaptation_callback=adaptation_callback
        )

        # Save any remaining ticks
        for tick in tick_data_buffer:
            tick_repo.save_tick(tick)

        # Add adaptation info to results
        if results and isinstance(results, dict):
            results['adaptation_enabled'] = True
            results['checkpoints'] = [130, 260]

        # Save results to database
        sim_repo.save_results(sim_id, results)

        # Store algorithms to MongoDB with their performance metrics
        if results and results.get('leaderboard'):
            print(f"\n💾 Saving algorithms to MongoDB...")
            generation = gen_repo.find_by_id(gen_id)
            if generation:
                algorithms_code = generation.get('algorithms', {})
                # Map agent names to their code
                algo_code_map = {}
                for model_name, code in algorithms_code.items():
                    # Create the sanitized agent name format
                    sanitized = model_name.replace('/', '_').replace('-', '_').replace(':', '_').replace('.', '_')
                    agent_name = f"generated_algo_{sanitized}"
                    algo_code_map[agent_name] = code
                
                saved_count = algo_repo.save_from_simulation_results(
                    simulation_id=sim_id,
                    generation_id=gen_id,
                    leaderboard=results['leaderboard'],
                    algorithms_code=algo_code_map
                )
                print(f"💾 Saved {saved_count} algorithms to MongoDB")

        print(f"✅ Adaptive simulation {sim_id} completed successfully")

    except Exception as e:
        print(f"❌ Adaptive simulation error: {e}")
        import traceback
        traceback.print_exc()
        sim_repo.set_error(sim_id, str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # Exclude generate_algo folder from file watcher to prevent restarts during simulation
    # The generate_algo folder contains generated algorithm files that change frequently
    extra_files = []
    exclude_patterns = ['*/generate_algo/*', '*/generate_algo/**']
    app.run(
        host="0.0.0.0",
        port=port,
        debug=True,
        use_reloader=True,
        extra_files=extra_files,
        exclude_patterns=exclude_patterns
    )
