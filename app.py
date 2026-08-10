import time
import numpy as np
from flask import Flask, jsonify, request, render_template
from numba import jit, prange

app = Flask(__name__)

# ============================================================================
# PIPELINE 1: EULERIAN CONTINUUM SOLVER (Grid-Based Stencils & PDEs)
# ============================================================================

@jit(nopython=True, parallel=True)
def solve_grayscott_grid(rows, cols, steps, du=0.2, dv=0.1, f=0.036, k=0.065, dt=1.0):
    """Numba-accelerated Gray-Scott Reaction-Diffusion Stencil Solver"""
    U = np.ones((rows, cols), dtype=np.float32)
    V = np.zeros((rows, cols), dtype=np.float32)
    
    # Seed a central chemical reaction core
    c_r, c_c = rows // 2, cols // 2
    r_size = int(rows * 0.15)
    U[c_r-r_size:c_r+r_size, c_c-r_size:c_c+r_size] = 0.5
    V[c_r-r_size:c_r+r_size, c_c-r_size:c_c+r_size] = 0.25
    
    # Add minor stochastic noise to break symmetry
    np.random.seed(42)
    for i in prange(rows):
        for j in prange(cols):
            U[i, j] += np.random.normal(0, 0.02)
            V[i, j] += np.random.normal(0, 0.02)

    history = []
    
    for s in range(steps):
        U_new = np.copy(U)
        V_new = np.copy(V)
        
        for i in prange(1, rows - 1):
            for j in prange(cols - 1):
                jp = (j + 1) % cols
                jm = (j - 1 + cols) % cols
                
                # 5-point Laplacian stencil
                lap_u = (U[i-1, j] + U[i+1, j] + U[i, jm] + U[i, jp] - 4.0 * U[i, j])
                lap_v = (V[i-1, j] + V[i+1, j] + V[i, jm] + V[i, jp] - 4.0 * V[i, j])
                
                uvv = U[i, j] * V[i, j] * V[i, j]
                
                U_new[i, j] = U[i, j] + (du * lap_u - uvv + f * (1.0 - U[i, j])) * dt
                V_new[i, j] = V[i, j] + (dv * lap_v + uvv - (f + k) * V[i, j]) * dt
                
        U, V = U_new, V_new
        
        # Capture frames for temporal playback history (10 frames)
        if s % max(1, steps // 10) == 0:
            history.append(V.flatten().astype(np.float64).tolist())
            
    return V, history

def run_eulerian_pipeline(engine, resolution, steps):
    """Dispatcher for grid-based continuum models"""
    start_time = time.time()
    
    if engine in ["grayscott", "bz_reaction"]:
        grid, history = solve_grayscott_grid(resolution, resolution, steps)
    else:
        grid = np.zeros((resolution, resolution), dtype=np.float32)
        history = []

    elapsed_ms = (time.time() - start_time) * 1000.0
    
    flat = grid.flatten()
    metrics = {
        "centroid_x": float(np.mean(flat)),
        "centroid_y": float(np.std(flat)),
        "std_x": float(np.max(flat)),
        "std_y": float(np.min(flat))
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
    """Existing Numba particle simulation kernel"""
    x = np.random.randn(walkers).astype(np.float32) * scale
    y = np.random.randn(walkers).astype(np.float32) * scale
    
    history_x = []
    history_y = []
    
    for s in range(steps):
        if engine_type == "lorenz":
            dx = 10.0 * (y - x)
            dy = x * 25.0 - y
            x += dx * 0.01
            y += dy * 0.01
        else:
            x += np.random.normal(0, 0.1, walkers).astype(np.float32) * scale
            y += np.random.normal(0, 0.1, walkers).astype(np.float32) * scale
            
        if s % max(1, steps // 10) == 0:
            history_x.append(x.copy())
            history_y.append(y.copy())
            
    return x, y, history_x, history_y

def run_lagrangian_pipeline(pane, resolution):
    """Dispatcher for particle/stochastic models"""
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
# UNIFIED API ROUTE DISPATCHER
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
        
        if engine in ["grayscott", "bz_reaction"]:
            logs.append(f"[BACKEND] Routing Pane #{idx+1} ({engine}) to Eulerian Grid Stencil Solver.")
            res_data = run_eulerian_pipeline(engine, resolution, pane["steps"])
        else:
            logs.append(f"[BACKEND] Routing Pane #{idx+1} ({engine}) to Stochastic Lagrangian Pipeline.")
            res_data = run_lagrangian_pipeline(pane, resolution)
            
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
