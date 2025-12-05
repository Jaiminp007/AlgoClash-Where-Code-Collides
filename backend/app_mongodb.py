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

# Initialize MongoDB on app startup
@app.before_first_request
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
            "filename": f"{ticker}_data.csv"
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
        stock = data.get('stock', 'AAPL_data.csv')

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
        if old_model in agents:
            idx = agents.index(old_model)
            agents[idx] = new_model

            # Update in database
            get_db().generations.update_one(
                {'generation_id': gen_id},
                {'$set': {'selected_models': agents}}
            )
            print(f"📝 Updated agents list: replaced position {idx}")

        # Remove old algorithm from generation.algorithms
        if old_model in generation.get("algorithms", {}):
            get_db().generations.update_one(
                {'generation_id': gen_id},
                {'$unset': {f'algorithms.{old_model}': ""}}
            )

        # Start regeneration in background thread
        thread = threading.Thread(
            target=regenerate_single_algorithm_background,
            args=(gen_id, new_model, generation.get("selected_stock", "AAPL_data.csv"))
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
    """Run simulation with pre-generated algorithms - MongoDB version"""
    try:
        generation = gen_repo.find_by_id(gen_id)

        if not generation:
            return jsonify({"error": "Generation not found"}), 404

        if generation["status"] != "completed":
            return jsonify({"error": "Algorithm generation not completed"}), 400

        # Generate unique simulation ID
        sim_id = str(uuid.uuid4())

        # Extract ticker from filename
        stock_file = generation["selected_stock"]
        ticker = stock_file.replace("_data.csv", "").upper()

        # Create Simulation document
        simulation = Simulation(
            simulation_id=sim_id,
            generation_id=gen_id,
            stock_ticker=ticker,
            selected_models=generation.get("selected_models", []),
            status="starting",
            progress=0,
            message="Initializing simulation..."
        )

        # Save to MongoDB
        sim_repo.create(simulation)

        # Start simulation in background thread
        thread = threading.Thread(
            target=run_simulation_only_background,
            args=(sim_id, gen_id, ticker, generation.get("selected_models", []))
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            "simulation_id": sim_id,
            "status": "started",
            "message": "Simulation started"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/api/simulation/<sim_id>")
def get_simulation_status(sim_id):
    """Get status of a running simulation - MongoDB version"""
    simulation = sim_repo.find_by_id(sim_id)

    if not simulation:
        return jsonify({"error": "Simulation not found"}), 404

    return jsonify(simulation)


# Background task functions

def run_generation_background(gen_id, agents, stock_file):
    """Generate algorithms in background thread - MongoDB version"""
    try:
        print(f"🔄 Starting generation for {len(agents)} agents in generation {gen_id}")

        # Update status
        gen_repo.update_status(gen_id, "generating", 10, "Generating algorithms...")

        # Import generation function
        from open_router.algo_gen import generate_algorithms_for_agents

        # Extract ticker from filename
        ticker = stock_file.replace("_data.csv", "").upper()
        print(f"📊 Using ticker: {ticker}")

        # Progress callback
        total_agents = len(agents)
        completed_count = [0]  # Using list for mutable counter in closure

        def progress_callback(progress, message):
            try:
                if isinstance(message, str) and message.startswith("PREVIEW::"):
                    parts = message.split("::", 2)
                    if len(parts) == 3:
                        _, model, code = parts
                        # Save algorithm to generation document
                        gen_repo.add_algorithm(gen_id, model, code)

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

            except Exception as e:
                print(f"Progress callback error: {e}")

        # Generate algorithms
        success = generate_algorithms_for_agents(agents, ticker, progress_callback)

        if success:
            gen_repo.update_status(gen_id, "completed", 100, "All algorithms generated successfully")
        else:
            gen_repo.set_error(gen_id, "Some algorithms failed to generate")

    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        gen_repo.set_error(gen_id, str(e))


def regenerate_single_algorithm_background(gen_id, model, stock_file):
    """Regenerate a single algorithm in background thread - MongoDB version"""
    try:
        print(f"🔄 Starting regeneration for {model} in generation {gen_id}")

        # Import generation function
        from open_router.algo_gen import generate_algorithms_for_agents

        # Extract ticker from filename
        ticker = stock_file.replace("_data.csv", "").upper()
        print(f"📊 Using ticker: {ticker}")

        # Progress callback for single model
        def progress_callback(progress, message):
            try:
                if isinstance(message, str) and message.startswith("PREVIEW::"):
                    parts = message.split("::", 2)
                    if len(parts) == 3:
                        _, msg_model, code = parts
                        if msg_model == model:
                            # Update generation document
                            gen_repo.add_algorithm(gen_id, model, code)

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
            except Exception as e:
                print(f"Progress callback error: {e}")

        # Generate algorithm for single model
        print(f"🤖 Calling generate_algorithms_for_agents for {model}...")
        success = generate_algorithms_for_agents([model], ticker, progress_callback)
        print(f"{'✅' if success else '❌'} Generation result for {model}: {success}")

    except Exception as e:
        print(f"❌ Regeneration error: {e}")
        import traceback
        traceback.print_exc()


def run_simulation_only_background(sim_id, gen_id, ticker, agents_list):
    """Run simulation with pre-generated algorithms - MongoDB version"""
    try:
        print(f"🚀 Starting simulation {sim_id} for {ticker}")

        # Update status
        sim_repo.update_status(sim_id, "running", 0, "Loading algorithms...")

        # Import simulation function
        from main import run_market_simulation

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

        # Run simulation
        results = run_market_simulation(
            ticker,
            progress_callback=progress_callback,
            tick_callback=tick_callback,
            allowed_models=agents_list
        )

        # Save any remaining ticks
        for tick in tick_data_buffer:
            tick_repo.save_tick(tick)

        # Save results to database
        sim_repo.save_results(sim_id, results)

        print(f"✅ Simulation {sim_id} completed successfully")

    except Exception as e:
        print(f"❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()
        sim_repo.set_error(sim_id, str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
