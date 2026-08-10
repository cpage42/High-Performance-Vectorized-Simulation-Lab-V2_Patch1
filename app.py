import time
import numpy as np
from flask import Flask, jsonify, request, render_template
from numba import jit, prange

app = Flask(__name__)

# ============================================================================
# PIPELINE 1: EULERIAN CONTINUUM SOLVER (Grid-Based Stencils & PDEs)
# ============================================================================

@jit(nopython=True, parallel=True)
def solve_grayscott_grid(rows, cols, steps, boundary="periodic", du=0.2, dv=0.1, f=0.036, k=0.065):
    dx = 1.0 / rows
    max_d = max(du, dv)
    dt = 0.4 * (dx ** 2) / max_d
    if dt > 1.0:
        dt = 1.0

    U = np.ones((rows, cols), dtype=np.float32)
    V = np.zeros((rows, cols), dtype=np.float32)
    
    c_r, c_c = rows // 2, cols // 2
    r_size = int(rows * 0.15)
    U[c_r-r_size:c_r+r_size, c_c-r_size:c_c+r_size] = 0.5
    V[c_r-r_size:c_r+r_size, c_c-r_size:c_c+r_size] = 0.25
    
    np.random.seed(42)
    for i in prange(rows):
        for j in prange(cols):
            U[i, j] += np.random.normal(0, 0.02)
            V[i, j] += np.random.normal(0, 0.02)

    # Pre-allocate history array for Numba compatibility (avoids .tolist() inside JIT)
    max_frames = 15
    history_arr = np.zeros((max_frames, rows, cols), dtype=np.float32)
    frame_count = 0
    interval = max(1, steps // 10)
    
    for s in range(steps):
        if s % interval == 0 and frame_count < max_frames:
            for ii in prange(rows):
                for jj in prange(cols):
                    history_arr[frame_count, ii, jj] = V[ii, jj]
            frame_count += 1

        U_new = np.copy(U)
        V_new = np.copy(V)
        
        for i in prange(1, rows - 1):
            for j in prange(1, cols - 1):
                if boundary == "periodic":
                    ip = (i + 1) % rows
                    im = (i - 1 + rows) % rows
                    jp = (j + 1) % cols
                    jm = (j - 1 + cols) % cols
                    lap_u = U[im, j] + U[ip, j] + U[i, jm] + U[i, jp] - 4.0 * U[i, j]
                    lap_v = V[im, j] + V[ip, j] + V[i, jm] + V[i, jp] - 4.0 * V[i, j]
                elif boundary == "reflective":
                    ip = i + 1 if i + 1 < rows else i
                    im = i - 1 if i - 1 >= 0 else i
                    jp = j + 1 if j + 1 < cols else j
                    jm = j - 1 if j - 1 >= 0 else j
                    lap_u = U[im, j] + U[ip, j] + U[i, jm] + U[i, jp] - 4.0 * U[i, j]
                    lap_v = V[im, j] + V[ip, j] + V[i, jm] + V[i, jp] - 4.0 * V[i, j]
                else:
                    ip = i + 1 if i + 1 < rows else rows - 1
                    im = i - 1 if i - 1 >= 0 else 0
                    jp = j + 1 if j + 1 < cols else cols - 1
                    jm = j - 1 if j - 1 >= 0 else 0
                    lap_u = U[im, j] + U[ip, j] + U[i, jm] + U[i, jp] - 4.0 * U[i, j]
                    lap_v = V[im, j] + V[ip, j] + V[i, jm] + V[i, jp] - 4.0 * V[i, j]
                
                uvv = U[i, j] * V[i, j] * V[i, j]
                U_new[i, j] = U[i, j] + (du * lap_u - uvv + f * (1.0 - U[i, j])) * dt
                V_new[i, j] = V[i, j] + (dv * lap_v + uvv - (f + k) * V[i, j]) * dt
                
        U, V = U_new, V_new
            
    return V, history_arr, frame_count

def run_eulerian_pipeline(engine, resolution, steps, boundary):
    start_time = time.time()
    
    if engine in ["grayscott", "bz_reaction"]:
        grid, hist_arr, frame_count = solve_grayscott_grid(resolution, resolution, steps, boundary)
        # Convert history array to Python list outside the JIT boundary
        history = [hist_arr[i].flatten().astype(np.float64).tolist() for i in range(frame_count)]
    else:
        grid = np.zeros((resolution, resolution), dtype=np.float32)
        history = []

    elapsed_ms = (time.time() - start_time) * 1000.0
    flat = grid.flatten()
    total_mass = float(np.sum(flat))
    peak_concentration = float(np.max(flat))
    
    y_indices, x_indices = np.indices(grid.shape)
    com_x = float(np.sum(x_indices * grid) / total_mass) if total_mass > 0 else 0.0
    com_y = float(np.sum(y_indices * grid) / total_mass) if total_mass > 0 else 0.0

    metrics = {
        "centroid_x": round(com_x, 2),
        "centroid_y": round(com_y, 2),
        "std_x": round(peak_concentration, 4),
        "std_y": round(total_mass, 2)
    }
    
    return {
        "density_grid": flat.astype(np.float64).tolist(),
        "history": history,
        "engine_time_ms": round(elapsed_ms, 2),
        "metrics": metrics
    }

# ============================================================================
# PIPELINE 2: STOCHASTIC LAGRANGIAN SOLVER (Particle Engines)
# ============================================================================

@jit(nopython=True)
def run_particle_simulation(walkers, steps, scale, engine_type):
    x = np.random.randn(walkers).astype(np.float32) * scale
    y = np.random.randn(walkers).astype(np.float32) * scale
    
    history_x = []
    history_y = []
    
    f32_10 = np.float32(10.0)
    f32_8 = np.float32(8.0)
    f32_dt_lorenz = np.float32(0.005)
    f32_dt_heston = np.float32(0.01)
    f32_zero5 = np.float32(0.05)
    
    for s in range(steps):
        if engine_type == "lorenz":
            dx = f32_10 * (y - x)
            dy = x * f32_8 - y
            x += dx * f32_dt_lorenz
            y += dy * f32_dt_lorenz
        elif engine_type in ["heston", "bates", "black_scholes"]:
            vol = np.abs(np.random.normal(0.2, 0.05, walkers)).astype(np.float32)
            x += (np.float32(0.05) - np.float32(0.5) * vol**2) * f32_dt_heston + vol * np.sqrt(f32_dt_heston) * np.random.randn(walkers).astype(np.float32)
            y += np.float32(-0.7) * vol * np.random.randn(walkers).astype(np.float32) * np.sqrt(f32_dt_heston)
        elif engine_type in ["schrodinger", "quantum"]:
            phase = np.float32(s * 0.15)
            x_old = x.copy()
            x = x_old * np.cos(phase) - y * np.sin(phase)
            y = x_old * np.sin(phase) + y * np.cos(phase)
            x += np.random.normal(0.0, 0.02, walkers).astype(np.float32)
        elif engine_type in ["thermal", "kramers"]:
            r = np.sqrt(x**2 + y**2) + np.float32(1e-5)
            x += (x / r) * f32_zero5 * scale + np.random.normal(0.0, 0.05, walkers).astype(np.float32)
            y += (y / r) * f32_zero5 * scale + np.random.normal(0.0, 0.05, walkers).astype(np.float32)
        elif engine_type == "navier_stokes":
            theta = np.float32(0.05)
            x_old = x.copy()
            x = x_old * np.cos(theta) - y * np.sin(theta)
            y = x_old * np.sin(theta) + y * np.cos(theta)
            x += np.random.normal(0.0, 0.08, walkers).astype(np.float32)
        else:
            x += np.random.normal(0.0, 0.1, walkers).astype(np.float32) * scale
            y += np.random.normal(0.0, 0.1, walkers).astype(np.float32) * scale
            
        if s % max(1, steps // 10) == 0:
            history_x.append(x.copy())
            history_y.append(y.copy())
            
    return x, y, history_x, history_y

def run_lagrangian_pipeline(pane, resolution):
    start_time = time.time()
    walkers = pane["walkers"]
    steps = pane["steps"]
    scale = pane["step_scale"]
    engine = pane["engine"]
    
    x, y, h_x, h_y = run_particle_simulation(walkers, steps, scale, engine)
    
    grid, _, _ = np.histogram2d(y, x, bins=resolution, range=[[-15, 15], [-15, 15]])
    grid = grid.astype(np.float32)
    
    history = []
    if pane.get("record_history", False):
        for i in range(len(h_x)):
            hg, _, _ = np.histogram2d(h_y[i], h_x[i], bins=resolution, range=[[-15, 15], [-15, 15]])
            history.append(hg.flatten().astype(np.float64).tolist())

    elapsed_ms = (time.time() - start_time) * 1000.0
    flat_grid = grid.flatten()
    
    metrics = {
        "centroid_x": round(float(np.mean(x)), 4),
        "centroid_y": round(float(np.mean(y)), 4),
        "std_x": round(float(np.std(x)), 4),
        "std_y": round(float(np.std(y)), 4)
    }
    
    return {
        "density_grid": flat_grid.astype(np.float64).tolist(),
        "history": history,
        "engine_time_ms": round(elapsed_ms, 2),
        "metrics": metrics
    }

# ============================================================================
# UNIFIED API ROUTE DISPATCHER WITH EXCEPTION GUARDING
# ============================================================================

@app.route("/run", methods=["POST"])
def run_batch():
    data = request.json
    resolution = data.get("resolution", 200)
    panes = data.get("panes", [])
    results = []
    logs = []

    for idx, pane in enumerate(panes):
        engine = pane["engine"]
        boundary = pane.get("boundary", "periodic")
        try:
            if engine in ["grayscott", "bz_reaction"]:
                logs.append(f"[BACKEND] Routing Pane #{idx+1} ({engine}) to Eulerian Grid Stencil Solver ({boundary}).")
                res_data = run_eulerian_pipeline(engine, resolution, pane["steps"], boundary)
            else:
                logs.append(f"[BACKEND] Routing Pane #{idx+1} ({engine}) to Stochastic Lagrangian Pipeline.")
                res_data = run_lagrangian_pipeline(pane, resolution)
        except Exception as err:
            logs.append(f"[CRITICAL ERROR] Pane #{idx+1} ({engine}) failed: {str(err)}")
            fallback_grid = np.zeros(resolution * resolution, dtype=np.float32).flatten().tolist()
            res_data = {
                "density_grid": fallback_grid,
                "history": [],
                "engine_time_ms": 0.0,
                "metrics": {"centroid_x": 0.0, "centroid_y": 0.0, "std_x": 0.0, "std_y": 0.0}
            }
        results.append(res_data)

    return jsonify({
        "status": "success",
        "resolution": resolution,
        "panes": results,
        "logs": logs
    })

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
