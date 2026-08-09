import os
import time
import numpy as np
from flask import Flask, render_template, request, jsonify
from pipeline import SimulationPipeline

app = Flask(__name__)

def get_system_ram_bytes():
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except Exception:
        return 8 * 1024 * 1024 * 1024  # Fallback 8GB

def calculate_capacity(ratio_percent):
    total_ram = get_system_ram_bytes()
    total_gb = round(total_ram / (1024 ** 3), 2)
    ratio = max(5.0, min(float(ratio_percent), 80.0))
    
    allocated_bytes = total_ram * (ratio / 100.0)
    allocated_mb = round(allocated_bytes / (1024 ** 2), 2)
    
    # 8 bytes per walker (float32 for X and float32 for Y)
    max_walkers = int(allocated_bytes / 8)
    max_walkers = max(200_000, min(max_walkers, 20_000_000))
    
    return total_gb, allocated_mb, max_walkers, ratio

# Initialize default capacity at 20%
total_gb, allocated_mb, MAX_WALKERS, current_ratio = calculate_capacity(20)
pipeline = SimulationPipeline(max_walkers=MAX_WALKERS)

global_x_buffer = np.zeros(MAX_WALKERS, dtype=np.float32)
global_y_buffer = np.zeros(MAX_WALKERS, dtype=np.float32)

def update_global_buffers(new_max):
    global global_x_buffer, global_y_buffer, MAX_WALKERS, pipeline
    if new_max != MAX_WALKERS:
        MAX_WALKERS = new_max
        global_x_buffer = np.zeros(MAX_WALKERS, dtype=np.float32)
        global_y_buffer = np.zeros(MAX_WALKERS, dtype=np.float32)
        pipeline = SimulationPipeline(max_walkers=MAX_WALKERS)

def engine_diffusion(x_buf, y_buf, steps, step_scale):
    n = len(x_buf)
    for _ in range(steps):
        x_buf += np.random.normal(0, step_scale, n)
        y_buf += np.random.normal(0, step_scale, n)

def engine_ou(x_buf, y_buf, steps, step_scale):
    n = len(x_buf)
    theta = 0.1
    for _ in range(steps):
        x_buf += -theta * x_buf + np.random.normal(0, step_scale, n)
        y_buf += -theta * y_buf + np.random.normal(0, step_scale, n)

def engine_drift(x_buf, y_buf, steps, step_scale):
    n = len(x_buf)
    for _ in range(steps):
        x_buf += step_scale
        y_buf += step_scale * 0.5

ENGINES = {
    "diffusion": engine_diffusion,
    "ou": engine_ou,
    "drift": engine_drift
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/hardware", methods=["GET"])
def hardware_info():
    ratio = request.args.get("ratio", 20)
    total_gb, allocated_mb, max_walkers, ratio = calculate_capacity(ratio)
    return jsonify({
        "total_ram_gb": total_gb,
        "allocated_mb": allocated_mb,
        "max_walkers": max_walkers,
        "ratio": ratio
    })

@app.route("/run", methods=["POST"])
def run_simulation():
    try:
        data = request.json
        ratio = data.get("ram_ratio", 20)
        total_gb, allocated_mb, max_walkers, ratio = calculate_capacity(ratio)
        
        # Dynamically resize buffers if user changed allocation ratio
        update_global_buffers(max_walkers)

        engine_choice = data.get("engine", "diffusion")
        walkers = int(data.get("walkers", 100000))
        
        if walkers > MAX_WALKERS:
            return jsonify({"status": "error", "message": f"Requested walkers ({walkers:,}) exceed your configured RAM allocation limit ({MAX_WALKERS:,}). Increase your RAM allocation ratio."})

        steps = int(data.get("steps", 100))
        step_scale = float(data.get("step_scale", 1.0))
        user_code_string = data.get("code", "")

        pipeline.clear_rules()
        jit_success = True

        if user_code_string.strip():
            local_namespace = {}
            exec(user_code_string, {"np": np}, local_namespace)
            if "custom_rule" in local_namespace:
                jit_success = pipeline.register_rule(local_namespace["custom_rule"])
            else:
                return jsonify({"status": "error", "message": "Your code must define a function named 'custom_rule(x, y)'."})

        active_x = global_x_buffer[:walkers]
        active_y = global_y_buffer[:walkers]
        active_x.fill(0.0)
        active_y.fill(0.0)

        selected_engine = ENGINES.get(engine_choice, engine_diffusion)
        engine_params = {"steps": steps, "step_scale": step_scale}
        
        final_x, final_y, engine_time, total_time = pipeline.execute_pipeline(active_x, active_y, selected_engine, engine_params)

        metrics = {
            "centroid_x": round(float(np.mean(final_x)), 4),
            "centroid_y": round(float(np.mean(final_y)), 4),
            "std_x": round(float(np.std(final_x)), 4),
            "std_y": round(float(np.std(final_y)), 4),
            "min_x": round(float(np.min(final_x)), 4),
            "max_x": round(float(np.max(final_x)), 4),
            "min_y": round(float(np.min(final_y)), 4),
            "max_y": round(float(np.max(final_y)), 4),
        }

        return jsonify({
            "status": "success", 
            "x": final_x.tolist(), 
            "y": final_y.tolist(),
            "jit_success": jit_success,
            "engine_time_ms": round(engine_time, 2),
            "total_time_ms": round(total_time, 2),
            "metrics": metrics
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
