import os
import json
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import shutil

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

# Register algorithm preview/management blueprint
from api.algos import algos_bp
app.register_blueprint(algos_bp)

# Health check
@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})

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
    """List available stock CSVs in backend/data ending with *_data.csv"""
    try:
        data_dir = Path(__file__).resolve().parent / "data"
        if not data_dir.exists():
            return jsonify({"stocks": []})
        files = sorted([p.name for p in data_dir.glob("*_data.csv")])
        stocks = [{
            "ticker": name.replace("_data.csv", "").upper(),
            "filename": name
        } for name in files]
        return jsonify({"stocks": stocks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Store running simulations and generations
running_simulations = {}
running_generations = {}  # Store algorithm generation sessions

@app.post("/api/run")
def run_simulation():
    """Start a new simulation with selected agents and stock"""
    try:
        data = request.get_json()
        agents = data.get('agents', [])
        stock = data.get('stock', 'AAPL_data.csv')
        
        # Validate we have at least 2 agents
        if len(agents) < 2:
            return jsonify({"error": "At least 2 agents are required"}), 400
            
        # Generate unique simulation ID
        sim_id = f"sim_{int(time.time())}"
        
        # Store simulation status
        running_simulations[sim_id] = {
            "status": "starting",
            "progress": 0,
            "results": None,
            "error": None
        }
        
        # Start simulation in background thread
        thread = threading.Thread(
            target=run_simulation_background,
            args=(sim_id, agents, stock)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "simulation_id": sim_id,
            "status": "started",
            "message": "Simulation started successfully"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/api/simulation/<sim_id>")
def get_simulation_status(sim_id):
    """Get status of a running simulation"""
    if sim_id not in running_simulations:
        return jsonify({"error": "Simulation not found"}), 404

    return jsonify(running_simulations[sim_id])

@app.post("/api/generate")
def generate_algorithms():
    """Generate algorithms only (no simulation)"""
    try:
        data = request.get_json()
        agents = data.get('agents', [])
        stock = data.get('stock', 'AAPL_data.csv')

        if len(agents) < 2:
            return jsonify({"error": "At least 2 agents are required"}), 400

        # Generate unique generation ID
        gen_id = f"gen_{int(time.time())}"

        # Store generation status
        running_generations[gen_id] = {
            "status": "starting",
            "progress": 0,
            "algorithms": {},
            "error": None,
            "agents": agents,
            "stock": stock
        }

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
    """Get status of algorithm generation"""
    if gen_id not in running_generations:
        return jsonify({"error": "Generation not found"}), 404

    return jsonify(running_generations[gen_id])

@app.route("/api/generation/<gen_id>/regenerate", methods=['POST'])
def regenerate_single_algorithm(gen_id):
    """Regenerate a single algorithm for a specific model"""
    try:
        print(f"\n{'='*60}")
        print(f"📥 REGENERATE REQUEST for generation: {gen_id}")

        if gen_id not in running_generations:
            print(f"❌ Generation {gen_id} not found")
            return jsonify({"error": "Generation not found"}), 404

        data = request.get_json()
        old_model = data.get('old_model')
        new_model = data.get('new_model')

        print(f"🔄 Replacing: {old_model} → {new_model}")

        if not old_model or not new_model:
            return jsonify({"error": "Both old_model and new_model are required"}), 400

        gen_data = running_generations[gen_id]

        # Update the agent list - replace only the FIRST occurrence to avoid
        # accidentally regenerating multiple slots with the same failed model
        agents = gen_data.get("agents", [])
        if old_model in agents:
            # Find the first index of old_model and replace only that one
            idx = agents.index(old_model)
            agents = agents[:idx] + [new_model] + agents[idx+1:]
            running_generations[gen_id]["agents"] = agents
            print(f"📝 Updated agents list: replaced position {idx}")
        else:
            print(f"⚠️ old_model '{old_model}' not found in agents list, skipping list update")

        # Mark the new model as generating
        model_states = gen_data.setdefault("model_states", {})
        model_states[new_model] = "generating"

        # Remove old algorithm and state if exists
        if old_model in gen_data.get("algorithms", {}):
            del gen_data["algorithms"][old_model]
        if old_model in model_states:
            del model_states[old_model]

        # Start regeneration in background thread
        thread = threading.Thread(
            target=regenerate_single_algorithm_background,
            args=(gen_id, new_model, gen_data.get("stock", "AAPL_data.csv"))
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
    """Run simulation with pre-generated algorithms"""
    try:
        if gen_id not in running_generations:
            return jsonify({"error": "Generation not found"}), 404

        gen_data = running_generations[gen_id]

        if gen_data["status"] != "completed":
            return jsonify({"error": "Algorithm generation not completed"}), 400

        # Generate unique simulation ID
        sim_id = f"sim_{int(time.time())}"

        # Store simulation status
        running_simulations[sim_id] = {
            "status": "starting",
            "progress": 0,
            "results": None,
            "error": None
        }

        # Extract ticker from filename
        stock_file = gen_data["stock"]
        ticker = stock_file.replace("_data.csv", "").upper()

        # Get the agents list from generation data
        agents_list = gen_data.get("agents", [])

        # Start simulation in background thread
        thread = threading.Thread(
            target=run_simulation_only_background,
            args=(sim_id, ticker, agents_list)
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

def regenerate_single_algorithm_background(gen_id, model, stock_file):
    """Regenerate a single algorithm in background thread"""
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
                state = running_generations[gen_id]
                model_states = state.setdefault("model_states", {})

                if isinstance(message, str) and message.startswith("PREVIEW::"):
                    parts = message.split("::", 2)
                    if len(parts) == 3:
                        _tag, msg_model, code = parts
                        if msg_model == model:
                            state["algorithms"][model] = code
                            model_states[model] = "done"
                elif isinstance(message, str) and message.startswith("MODEL_OK::"):
                    parts = message.split("::", 2)
                    if len(parts) >= 2 and parts[1] == model:
                        model_states[model] = "done"
                elif isinstance(message, str) and message.startswith("MODEL_FAIL::"):
                    parts = message.split("::", 2)
                    if len(parts) >= 2 and parts[1] == model:
                        model_states[model] = "error"
                elif isinstance(message, str) and message.startswith("MODEL_START::"):
                    parts = message.split("::", 2)
                    if len(parts) >= 2 and parts[1] == model:
                        model_states[model] = "generating"
            except Exception as e:
                print(f"Progress callback error: {e}")

        # Generate algorithm for single model
        print(f"🤖 Calling generate_algorithms_for_agents for {model}...")
        success = generate_algorithms_for_agents([model], ticker, progress_callback)
        print(f"{'✅' if success else '❌'} Generation result for {model}: {success}")

        if success:
            # Read generated file
            try:
                from pathlib import Path
                backend_root = Path(__file__).resolve().parent
                gen_dir = backend_root / "generate_algo"

                def _sanitize(name: str) -> str:
                    return name.replace('/', '_').replace('-', '_').replace(':', '_').replace('.', '_')

                sanitized = _sanitize(model)
                file_path = gen_dir / f"generated_algo_{sanitized}.py"

                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    running_generations[gen_id]["algorithms"][model] = content
                    running_generations[gen_id]["model_states"][model] = "done"
                    print(f"✅ Successfully loaded algorithm for {model} ({len(content)} chars)")
                else:
                    print(f"❌ Algorithm file not found for {model}: {file_path}")
                    running_generations[gen_id]["model_states"][model] = "error"
            except Exception as e:
                print(f"Failed reading regenerated algorithm: {e}")
                running_generations[gen_id]["model_states"][model] = "error"
        else:
            running_generations[gen_id]["model_states"][model] = "error"

    except Exception as e:
        print(f"Regeneration error: {e}")
        running_generations[gen_id]["model_states"][model] = "error"
        import traceback
        traceback.print_exc()

def run_generation_background(gen_id, agents, stock_file):
    """Generate algorithms in background thread"""
    try:
        # Clean up old generated algorithms first
        backend_root = Path(__file__).resolve().parent
        gen_dir = backend_root / "generate_algo"
        if gen_dir.exists() and gen_dir.is_dir():
            shutil.rmtree(gen_dir, ignore_errors=True)
            print(f"🧹 Cleaned up old algorithms before generation {gen_id}")

        running_generations[gen_id]["status"] = "running"
        running_generations[gen_id]["progress"] = 10
        running_generations[gen_id]["message"] = "Starting algorithm generation..."

        # Import generation function
        from open_router.algo_gen import generate_algorithms_for_agents

        # Extract ticker from filename
        ticker = stock_file.replace("_data.csv", "").upper()

        running_generations[gen_id]["progress"] = 20
        running_generations[gen_id]["message"] = f"Generating algorithms for {ticker}..."

        # Progress callback to store algorithm code and per-model states
        def progress_callback(progress, message):
            try:
                state = running_generations[gen_id]
                # Initialize collections if missing
                model_logs = state.setdefault("model_logs", [])
                model_states = state.setdefault("model_states", {})

                if isinstance(message, str) and message.startswith("PREVIEW::"):
                    parts = message.split("::", 2)
                    if len(parts) == 3:
                        _tag, model, code = parts
                        state["algorithms"][model] = code
                        state["message"] = f"Generated algorithm for {model}"
                elif isinstance(message, str) and (
                    message.startswith("MODEL_OK::") or 
                    message.startswith("MODEL_FAIL::") or 
                    message.startswith("MODEL_SKIP::") or 
                    message.startswith("MODEL_START::")
                ):
                    # Track per-model lifecycle events
                    model_logs.append(message)
                    try:
                        tag, model, *rest = message.split("::", 2)
                    except Exception:
                        tag, model = message, ""
                    # Normalize states for UI
                    if message.startswith("MODEL_START::"):
                        model_states[model] = "generating"
                    elif message.startswith("MODEL_OK::"):
                        model_states[model] = "done"
                    elif message.startswith("MODEL_FAIL::"):
                        model_states[model] = "error"
                    elif message.startswith("MODEL_SKIP::"):
                        model_states[model] = "skipped"
                    # Do not overwrite human-readable message unless none provided
                    state.setdefault("message", message)
                else:
                    state["message"] = message

                state["progress"] = progress
            except Exception as _:
                running_generations[gen_id]["progress"] = progress
                running_generations[gen_id]["message"] = str(message)

        # Generate algorithms
        success = generate_algorithms_for_agents(agents, ticker, progress_callback)

        if success:
            # On success, populate full algorithm contents from saved files for the UI
            try:
                backend_root = Path(__file__).resolve().parent
                gen_dir = backend_root / "generate_algo"
                algo_map = running_generations[gen_id].setdefault("algorithms", {})

                def _sanitize(name: str) -> str:
                    return name.replace('/', '_').replace('-', '_').replace(':', '_').replace('.', '_')

                if gen_dir.exists():
                    # Only load algorithms for the selected agents (not old files)
                    for file_path in gen_dir.glob('generated_algo_*.py'):
                        try:
                            filename = file_path.name
                            model_part = filename.replace('generated_algo_', '').replace('.py', '')
                            # Map back to original agent id if possible
                            orig = next((a for a in agents if _sanitize(a) == model_part), None)
                            # Only include if this file matches one of our selected agents
                            if orig:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                algo_map[orig] = content
                                print(f"✓ Loaded algorithm for {orig}")
                            else:
                                print(f"⚠️ Skipping old algorithm file: {filename} (not in selected agents)")
                        except Exception as fe:
                            print(f"Failed reading algorithm file {file_path}: {fe}")
            except Exception as pe:
                print(f"Post-process population of algorithms failed: {pe}")

            running_generations[gen_id]["status"] = "completed"
            running_generations[gen_id]["progress"] = 100
            running_generations[gen_id]["message"] = "All algorithms generated!"
        else:
            raise RuntimeError("Algorithm generation failed")

    except Exception as e:
        running_generations[gen_id]["status"] = "error"
        running_generations[gen_id]["error"] = str(e)
        running_generations[gen_id]["message"] = f"Error: {str(e)}"
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()

def run_simulation_only_background(sim_id, ticker, agents_list=None):
    """Run simulation only (algorithms already generated)"""
    try:
        running_simulations[sim_id]["status"] = "running"
        running_simulations[sim_id]["progress"] = 10
        running_simulations[sim_id]["message"] = "Starting market simulation..."

        # Import simulation function
        from main import run_market_simulation

        # Progress callback
        def progress_callback(progress, message):
            try:
                running_simulations[sim_id]["message"] = message
                running_simulations[sim_id]["progress"] = progress
            except Exception as _:
                running_simulations[sim_id]["progress"] = progress
                running_simulations[sim_id]["message"] = str(message)

        # Tick callback for real-time chart data
        def tick_callback(tick_num, tick_data, trades):
            try:
                if "chart_data" not in running_simulations[sim_id]:
                    running_simulations[sim_id]["chart_data"] = []

                # Convert trades to serializable format
                serialized_trades = []
                for trade in trades[-10:]:  # Keep last 10 trades per tick
                    try:
                        serialized_trades.append({
                            'agent': trade.agent_name if hasattr(trade, 'agent_name') else '',
                            'side': trade.side.value if hasattr(trade.side, 'value') else str(trade.side),
                            'quantity': trade.quantity if hasattr(trade, 'quantity') else 0,
                            'price': trade.price if hasattr(trade, 'price') else 0
                        })
                    except Exception:
                        pass

                running_simulations[sim_id]["chart_data"].append({
                    'tick': tick_num,
                    'price': tick_data.get('price', 0),
                    'timestamp': str(tick_data.get('timestamp', '')),
                    'trades': serialized_trades
                })

                # Keep only last 100 ticks to avoid memory issues
                if len(running_simulations[sim_id]["chart_data"]) > 100:
                    running_simulations[sim_id]["chart_data"] = running_simulations[sim_id]["chart_data"][-100:]

            except Exception as e:
                print(f"Tick callback error: {e}")

        # Use the agents list passed from the generation data
        allowed = agents_list if agents_list else None

        print(f"🎯 Running simulation with {len(allowed) if allowed else 'all'} agents: {allowed}")

        # Run simulation with allowed models if available
        results = run_market_simulation(ticker, progress_callback, allowed_models=allowed, tick_callback=tick_callback)

        running_simulations[sim_id]["status"] = "completed"
        running_simulations[sim_id]["progress"] = 100
        running_simulations[sim_id]["message"] = "Simulation completed!"
        running_simulations[sim_id]["results"] = results

    except Exception as e:
        running_simulations[sim_id]["status"] = "error"
        running_simulations[sim_id]["error"] = str(e)
        running_simulations[sim_id]["message"] = f"Error: {str(e)}"
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup: remove generated algorithms after simulation
        try:
            backend_root = Path(__file__).resolve().parent
            gen_dir = backend_root / "generate_algo"
            if gen_dir.exists() and gen_dir.is_dir():
                shutil.rmtree(gen_dir, ignore_errors=True)
                print("🧹 Cleaned up generated algorithms folder.")
        except Exception as ce:
            print(f"Cleanup error: {ce}")

def run_simulation_background(sim_id, agents, stock_file):
    """Run the simulation in background thread"""
    try:
        running_simulations[sim_id]["status"] = "running"
        running_simulations[sim_id]["progress"] = 10
        running_simulations[sim_id]["message"] = "Starting simulation..."

        # Import and run the main simulation logic
        from main import run_simulation_with_params

        # Extract ticker from filename
        ticker = stock_file.replace("_data.csv", "").upper()

        running_simulations[sim_id]["progress"] = 20
        running_simulations[sim_id]["message"] = f"Generating algorithms for {ticker}..."

        # Run the simulation with progress callback
        def progress_callback(progress, message):
            # Support special preview messages: "PREVIEW::<model>::<code>"
            try:
                if isinstance(message, str) and message.startswith("PREVIEW::"):
                    parts = message.split("::", 2)
                    if len(parts) == 3:
                        _tag, model, code = parts
                        running_simulations[sim_id]["preview_model"] = model
                        running_simulations[sim_id]["code_preview"] = code
                elif isinstance(message, str) and (message.startswith("MODEL_OK::") or message.startswith("MODEL_FAIL::") or message.startswith("MODEL_SKIP::")):
                    # Forward per-model event logs for UI
                    # Format: TAG::model::reason
                    logs = running_simulations[sim_id].setdefault("model_logs", [])
                    logs.append(message)
                else:
                    running_simulations[sim_id]["message"] = message
                running_simulations[sim_id]["progress"] = progress
            except Exception as _:
                running_simulations[sim_id]["progress"] = progress
                running_simulations[sim_id]["message"] = str(message)

        results = run_simulation_with_params(agents, ticker, progress_callback)

        running_simulations[sim_id]["status"] = "completed"
        running_simulations[sim_id]["progress"] = 100
        running_simulations[sim_id]["message"] = "Simulation completed!"
        running_simulations[sim_id]["results"] = results

    except Exception as e:
        running_simulations[sim_id]["status"] = "error"
        running_simulations[sim_id]["error"] = str(e)
        running_simulations[sim_id]["message"] = f"Error: {str(e)}"
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup: remove generated algorithms after each run
        try:
            backend_root = Path(__file__).resolve().parent
            gen_dir = backend_root / "generate_algo"
            if gen_dir.exists() and gen_dir.is_dir():
                shutil.rmtree(gen_dir, ignore_errors=True)
                print("🧹 Cleaned up generated algorithms folder.")
                running_simulations[sim_id]["message"] = (running_simulations[sim_id].get("message") or "") + "\n🧹 Cleaned generated algorithms."
        except Exception as ce:
            # Log cleanup failure but do not crash
            print(f"Cleanup error: {ce}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
