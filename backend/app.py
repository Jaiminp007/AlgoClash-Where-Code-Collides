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

# Store running simulations
running_simulations = {}

@app.post("/api/run")
def run_simulation():
    """Start a new simulation with selected agents and stock"""
    try:
        data = request.get_json()
        agents = data.get('agents', [])
        stock = data.get('stock', 'AAPL_data.csv')
        # Optional aggressiveness/volume control (default 1.0)
        try:
            volume_multiplier = float(data.get('volume_multiplier', 1.0))
        except Exception:
            volume_multiplier = 1.0
        # Aggressive mode can be explicitly set by client; if not, infer from high volume
        aggressive_mode = bool(data.get('aggressive_mode', False)) or (volume_multiplier >= 2.5)
        
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
            args=(sim_id, agents, stock, volume_multiplier, aggressive_mode)
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

def run_simulation_background(sim_id, agents, stock_file, volume_multiplier: float = 1.0, aggressive_mode: bool = False):
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
                else:
                    running_simulations[sim_id]["message"] = message
                running_simulations[sim_id]["progress"] = progress
            except Exception as _:
                running_simulations[sim_id]["progress"] = progress
                running_simulations[sim_id]["message"] = str(message)
        
        options = {"volume_multiplier": float(volume_multiplier), "aggressive_mode": bool(aggressive_mode)}
        results = run_simulation_with_params(agents, ticker, progress_callback, options)
        
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
