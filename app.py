import os
import time
import numpy as np
import json
import traceback
from flask import Flask, render_template, request, jsonify, Response
from pipeline import SimulationPipeline
from numba import njit

app = Flask(__name__)

def get_system_ram_bytes():
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except Exception:
        return 8 * 1024 * 1024 * 1024  

def calculate_capacity(ratio_percent):
    total_ram = get_system_ram_bytes()
    total_gb = round(total_ram / (1024 ** 3), 2)
    ratio = max(5.0, min(float(ratio_percent), 80.0))
    allocated_bytes = total_ram * (ratio / 100.0)
    allocated_mb = round(allocated_bytes / (1024 ** 2), 2)
    max_walkers = int(allocated_bytes / 8)
    max_walkers = max(200_000, min(max_walkers, 50_000_000))
    return total_gb, allocated_mb, max_walkers, ratio

total_gb, allocated_mb, MAX_WALKERS, current_ratio = calculate_capacity(20)
pipeline = SimulationPipeline(max_walkers=MAX_WALKERS)

global_x_buffer = np.zeros(MAX_WALKERS, dtype=np.float32)
global_y_buffer = np.zeros(MAX_WALKERS, dtype=np.float32)
global_last_x = np.array([])
global_last_y = np.array([])

def update_global_buffers(new_max):
    global global_x_buffer, global_y_buffer, MAX_WALKERS, pipeline
    if new_max != MAX_WALKERS:
        MAX_WALKERS = new_max
        global_x_buffer = np.zeros(MAX_WALKERS, dtype=np.float32)
        global_y_buffer = np.zeros(MAX_WALKERS, dtype=np.float32)
        pipeline = SimulationPipeline(max_walkers=MAX_WALKERS)

# --- Core JIT Engines (Physics & Chemistry Specialized) ---

@njit(fastmath=True, cache=True)
def engine_diffusion_bounded(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    box = 25.0 
    for _ in range(steps):
        for i in range(n):
            x_buf[i] += np.random.normal(0.0, step_scale)
            y_buf[i] += np.random.normal(0.0, step_scale)
            if boundary_mode == 1: 
                if x_buf[i] < -box: x_buf[i] = -box + (-box - x_buf[i])
                elif x_buf[i] > box: x_buf[i] = box - (x_buf[i] - box)
                if y_buf[i] < -box: y_buf[i] = -box + (-box - y_buf[i])
                elif y_buf[i] > box: y_buf[i] = box - (y_buf[i] - box)
            elif boundary_mode == 2: 
                if x_buf[i] < -box: x_buf[i] = box
                elif x_buf[i] > box: x_buf[i] = -box
                if y_buf[i] < -box: y_buf[i] = box
                elif y_buf[i] > box: y_buf[i] = -box

@njit(fastmath=True, cache=True)
def engine_ou_bounded(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    theta = 0.1
    box = 25.0
    for _ in range(steps):
        for i in range(n):
            x_buf[i] += -theta * x_buf[i] + np.random.normal(0.0, step_scale)
            y_buf[i] += -theta * y_buf[i] + np.random.normal(0.0, step_scale)
            if boundary_mode == 1:
                if x_buf[i] < -box: x_buf[i] = -box
                elif x_buf[i] > box: x_buf[i] = box
                if y_buf[i] < -box: y_buf[i] = -box
                elif y_buf[i] > box: y_buf[i] = box
            elif boundary_mode == 2:
                if x_buf[i] < -box: x_buf[i] = box
                elif x_buf[i] > box: x_buf[i] = -box
                if y_buf[i] < -box: y_buf[i] = box
                elif y_buf[i] > box: y_buf[i] = -box

@njit(fastmath=True, cache=True)
def engine_langevin(x_buf, y_buf, steps, step_scale, boundary_mode):
    """Langevin dynamics: Damped motion in a harmonic potential well with thermal noise."""
    n = len(x_buf)
    gamma = 0.05  # Damping coefficient
    k_trap = 0.02 # Trap stiffness
    box = 25.0
    for _ in range(steps):
        for i in range(n):
            fx = -k_trap * x_buf[i] - gamma * x_buf[i]
            fy = -k_trap * y_buf[i] - gamma * y_buf[i]
            x_buf[i] += fx + np.random.normal(0.0, step_scale)
            y_buf[i] += fy + np.random.normal(0.0, step_scale)
            if boundary_mode == 1:
                if x_buf[i] < -box: x_buf[i] = -box
                elif x_buf[i] > box: x_buf[i] = box
                if y_buf[i] < -box: y_buf[i] = -box
                elif y_buf[i] > box: y_buf[i] = box

@njit(fastmath=True, cache=True)
def engine_thermal_conduction(x_buf, y_buf, steps, step_scale, boundary_mode):
    """Simulates thermal diffusion gradients radiating from a high-energy central source."""
    n = len(x_buf)
    box = 25.0
    for _ in range(steps):
        for i in range(n):
            r = np.sqrt(x_buf[i]**2 + y_buf[i]**2) + 1e-4
            radial_push = 0.05 / r
            x_buf[i] += (x_buf[i] / r) * radial_push + np.random.normal(0.0, step_scale * 1.2)
            y_buf[i] += (y_buf[i] / r) * radial_push + np.random.normal(0.0, step_scale * 1.2)
            if boundary_mode == 1:
                if x_buf[i] < -box: x_buf[i] = -box
                elif x_buf[i] > box: x_buf[i] = box
                if y_buf[i] < -box: y_buf[i] = -box
                elif y_buf[i] > box: y_buf[i] = box

@njit(fastmath=True, cache=True)
def engine_chemical_kinetics(x_buf, y_buf, steps, step_scale, boundary_mode):
    """Lotka-Volterra inspired chemical species concentration phase space simulation."""
    n = len(x_buf)
    alpha, beta, delta, gamma_val = 1.1, 0.4, 0.1, 0.4
    box = 25.0
    for _ in range(steps):
        for i in range(n):
            u = x_buf[i]
            v = y_buf[i]
            du = (alpha * u - beta * u * v) * 0.05
            dv = (delta * u * v - gamma_val * v) * 0.05
            x_buf[i] += du + np.random.normal(0.0, step_scale * 0.2)
            y_buf[i] += dv + np.random.normal(0.0, step_scale * 0.2)
            if boundary_mode == 1:
                if x_buf[i] < -box: x_buf[i] = -box
                elif x_buf[i] > box: x_buf[i] = box
                if y_buf[i] < -box: y_buf[i] = -box
                elif y_buf[i] > box: y_buf[i] = box

@njit(fastmath=True, cache=True)
def fast_histogram2d(x, y, bins, x_min, x_max, y_min, y_max):
    grid = np.zeros((bins, bins), dtype=np.float32)
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range == 0: x_range = 1e-5
    if y_range == 0: y_range = 1e-5
    x_scale = bins / x_range
    y_scale = bins / y_range
    n = len(x)
    for i in range(n):
        xi = int((x[i] - x_min) * x_scale)
        yi = int((y[i] - y_min) * y_scale)
        if xi < 0: xi = 0
        elif xi >= bins: xi = bins - 1
        if yi < 0: yi = 0
        elif yi >= bins: yi = bins - 1
        grid[yi, xi] += 1.0
    return grid

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/hardware", methods=["GET"])
def hardware_info():
    ratio = request.args.get("ratio", 20)
    t_gb, a_mb, max_w, r = calculate_capacity(ratio)
    return jsonify({"total_ram_gb": t_gb, "allocated_mb": a_mb, "max_walkers": max_w, "ratio": r})

@app.route("/export", methods=["GET"])
def export_csv():
    def generate():
        yield "x,y\n"
        chunk_size = 100000
        for i in range(0, len(global_last_x), chunk_size):
            x_chunk = global_last_x[i:i+chunk_size]
            y_chunk = global_last_y[i:i+chunk_size]
            lines = [f"{x},{y}" for x, y in zip(x_chunk, y_chunk)]
            yield "\n".join(lines) + "\n"
    return Response(generate(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=simulation_coordinates.csv"})

@app.route("/run", methods=["POST"])
def run_simulation():
    global global_last_x, global_last_y
    try:
        data = request.json
        ratio = data.get("ram_ratio", 20)
        _, _, max_w, _ = calculate_capacity(ratio)
        update_global_buffers(max_w)

        panes = data.get("panes", [])
        res = int(data.get("resolution", 200))
        pane_results = []
        console_logs = []

        start_total = time.perf_counter()

        for idx, pane in enumerate(panes):
            engine_choice = pane.get("engine", "diffusion")
            walkers = int(pane.get("walkers", 500000))
            steps = int(pane.get("steps", 100))
            step_scale = float(pane.get("step_scale", 1.0))
            boundary_str = pane.get("boundary", "free")
            boundary_mode = 0 if boundary_str == "free" else (1 if boundary_str == "reflective" else 2)
            record_history = pane.get("record_history", False)

            if walkers > MAX_WALKERS:
                return jsonify({"status": "error", "message": f"Pane {idx+1}: Walkers exceed RAM allocation."})

            active_x = global_x_buffer[:walkers]
            active_y = global_y_buffer[:walkers]
            active_x.fill(0.0)
            active_y.fill(0.0)

            history_grids = []
            history_steps = min(steps, 10) if record_history else 1
            chunk_steps = max(1, steps // history_steps)

            t0 = time.perf_counter()
            for s in range(history_steps):
                cur_steps = chunk_steps if s < history_steps - 1 else steps - (chunk_steps * (history_steps - 1))
                if cur_steps <= 0: break
                
                if engine_choice == "ou":
                    engine_ou_bounded(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "langevin":
                    engine_langevin(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "thermal":
                    engine_thermal_conduction(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "kinetics":
                    engine_chemical_kinetics(active_x, active_y, cur_steps, step_scale, boundary_mode)
                else:
                    engine_diffusion_bounded(active_x, active_y, cur_steps, step_scale, boundary_mode)

                if record_history:
                    min_x, max_x = float(np.min(active_x)), float(np.max(active_x))
                    min_y, max_y = float(np.min(active_y)), float(np.max(active_y))
                    if min_x == max_x: max_x += 1e-5
                    if min_y == max_y: max_y += 1e-5
                    h_grid = fast_histogram2d(active_x, active_y, res, min_x, max_x, min_y, max_y)
                    history_grids.append(h_grid.flatten().tolist())

            t1 = time.perf_counter()
            engine_time = (t1 - t0) * 1000.0

            if idx == len(panes) - 1:
                global_last_x = active_x.copy()
                global_last_y = active_y.copy()

            metrics = {
                "centroid_x": round(float(np.mean(active_x)), 4),
                "centroid_y": round(float(np.mean(active_y)), 4),
                "std_x": round(float(np.std(active_x)), 4),
                "std_y": round(float(np.std(active_y)), 4),
                "min_x": round(float(np.min(active_x)), 4),
                "max_x": round(float(np.max(active_x)), 4),
                "min_y": round(float(np.min(active_y)), 4),
                "max_y": round(float(np.max(active_y)), 4),
            }

            x_min, x_max = float(metrics["min_x"]), float(metrics["max_x"])
            y_min, y_max = float(metrics["min_y"]), float(metrics["max_y"])
            if x_min == x_max: x_max += 1e-5
            if y_min == y_max: y_max += 1e-5

            final_heatmap = fast_histogram2d(active_x, active_y, res, x_min, x_max, y_min, y_max)

            pane_results.append({
                "pane_id": idx,
                "density_grid": final_heatmap.flatten().tolist(),
                "history": history_grids if record_history else [],
                "engine_time_ms": round(engine_time, 2),
                "jit_success": True,
                "metrics": metrics
            })
            console_logs.append(f"[INFO] Pane {idx+1}: {engine_choice} ({walkers:,} walkers) completed in {round(engine_time,2)}ms")

        end_total = time.perf_counter()
        console_logs.append(f"[SUCCESS] Batch execution finished in {round((end_total - start_total)*1000.0, 2)} ms.")

        return jsonify({
            "status": "success",
            "resolution": res,
            "panes": pane_results,
            "logs": console_logs
        })
    except Exception as e:
        err_msg = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "trace": err_msg})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)
