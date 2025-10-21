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

        # Start simulation in background thread
        thread = threading.Thread(
            target=run_simulation_only_background,
            args=(sim_id, ticker)
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

def run_generation_background(gen_id, agents, stock_file):
    """Generate algorithms in background thread"""
    try:
        running_generations[gen_id]["status"] = "running"
        running_generations[gen_id]["progress"] = 10
        running_generations[gen_id]["message"] = "Starting algorithm generation..."

        # Import generation function
        from open_router.algo_gen import generate_algorithms_for_agents

        # Extract ticker from filename
        ticker = stock_file.replace("_data.csv", "").upper()

        running_generations[gen_id]["progress"] = 20
        running_generations[gen_id]["message"] = f"Generating algorithms for {ticker}..."

        # Progress callback to store algorithm code
        def progress_callback(progress, message):
            try:
                if isinstance(message, str) and message.startswith("PREVIEW::"):
                    parts = message.split("::", 2)
                    if len(parts) == 3:
                        _tag, model, code = parts
                        running_generations[gen_id]["algorithms"][model] = code
                        running_generations[gen_id]["message"] = f"Generated algorithm for {model}"
                else:
                    running_generations[gen_id]["message"] = message
                running_generations[gen_id]["progress"] = progress
            except Exception as _:
                running_generations[gen_id]["progress"] = progress
                running_generations[gen_id]["message"] = str(message)

        # Generate algorithms
        success = generate_algorithms_for_agents(agents, ticker, progress_callback)

        if success:
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

def run_simulation_only_background(sim_id, ticker):
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

        # Run simulation
        results = run_market_simulation(ticker, progress_callback)

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
