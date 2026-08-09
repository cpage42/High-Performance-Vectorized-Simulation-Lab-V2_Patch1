import os

os.makedirs("templates", exist_ok=True)

with open("jit_compiler.py", "w") as f:
    f.write('''import numba

def auto_jit_rule(user_func):
    try:
        compiled_func = numba.njit(user_func, fastmath=True)
        return compiled_func, True
    except Exception as e:
        print(f"JIT compilation warning: executing as standard Python. ({e})")
        return user_func, False
''')

with open("pipeline.py", "w") as f:
    f.write('''import numpy as np
from jit_compiler import auto_jit_rule

class SimulationPipeline:
    def __init__(self, max_walkers: int):
        self.max_walkers = max_walkers
        self.rules = []

    def register_rule(self, user_func):
        compiled_rule, success = auto_jit_rule(user_func)
        self.rules.append(compiled_rule)
        return success

    def clear_rules(self):
        self.rules = []

    def execute_pipeline(self, x_buffer, y_buffer, engine_runner_func, engine_params):
        engine_runner_func(x_buffer, y_buffer, **engine_params)
        x = np.frombuffer(x_buffer, dtype=np.float32, count=self.max_walkers)
        y = np.frombuffer(y_buffer, dtype=np.float32, count=self.max_walkers)
        for rule in self.rules:
            x, y = rule(x, y)
        return x, y
''')

with open("templates/index.html", "w") as f:
    f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>High-Performance Simulation Sandbox</title>
    <style>
        body { font-family: monospace; background: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
        .container { display: flex; gap: 20px; }
        .controls { flex: 1; background: #1e1e1e; padding: 20px; border-radius: 8px; }
        .viewport { flex: 2; background: #1e1e1e; padding: 20px; border-radius: 8px; display: flex; flex-direction: column; align-items: center; }
        input, select, textarea, button { width: 100%; margin-bottom: 10px; background: #2d2d2d; color: #fff; border: 1px solid #444; padding: 8px; border-radius: 4px; box-sizing: border-box; }
        textarea { height: 160px; font-family: monospace; }
        button { background: #00bcd4; color: #000; font-weight: bold; cursor: pointer; }
        button:hover { background: #00acc1; }
        canvas { background: #000; border: 1px solid #333; width: 100%; max-width: 600px; height: 600px; }
    </style>
</head>
<body>
    <h1>Vectorized Simulation Sandbox</h1>
    <div class="container">
        <div class="controls">
            <h3>Select Core Engine</h3>
            <select id="engine_type">
                <option value="diffusion">Brownian Diffusion / Random Walk</option>
                <option value="ou">Ornstein-Uhlenbeck (Mean-Reverting)</option>
                <option value="drift">Linear Vector Drift / Flow</option>
            </select>
            <h3>Engine Parameters</h3>
            <label>Walkers / Particles:</label>
            <input type="number" id="walkers" value="100000">
            <label>Steps:</label>
            <input type="number" id="steps" value="100">
            <label>Step Scale / Variance:</label>
            <input type="number" step="0.1" id="step_scale" value="1.0">
            <h3>Custom Python Rule (Optional)</h3>
            <textarea id="user_code" placeholder="# Write your custom function here:
# def custom_rule(x, y):
#     return x, y"></textarea>
            <button onclick="runSimulation()">Execute Simulation</button>
        </div>
        <div class="viewport">
            <h3>Real-Time Output Viewport</h3>
            <canvas id="simCanvas" width="600" height="600"></canvas>
        </div>
    </div>
    <script>
        const canvas = document.getElementById("simCanvas");
        const ctx = canvas.getContext("2d");
        function drawPoints(xArr, yArr) {
            ctx.fillStyle = "#000";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "#00bcd4";
            for (let i = 0; i < xArr.length; i += 10) {
                let screenX = (xArr[i] + 100) * (canvas.width / 200);
                let screenY = (yArr[i] + 100) * (canvas.height / 200);
                ctx.fillRect(screenX, screenY, 2, 2);
            }
        }
        async function runSimulation() {
            const payload = {
                engine: document.getElementById("engine_type").value,
                walkers: parseInt(document.getElementById("walkers").value),
                steps: parseInt(document.getElementById("steps").value),
                step_scale: parseFloat(document.getElementById("step_scale").value),
                code: document.getElementById("user_code").value
            };
            const response = await fetch("/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            if (data.status === "success") { drawPoints(data.x, data.y); }
            else { alert("Error: " + data.message); }
        }
    </script>
</body>
</html>''')

with open("app.py", "w") as f:
    f.write('''import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from pipeline import SimulationPipeline

app = Flask(__name__)
MAX_WALKERS = 2_000_000
pipeline = SimulationPipeline(max_walkers=MAX_WALKERS)

global_x_buffer = np.zeros(MAX_WALKERS, dtype=np.float32)
global_y_buffer = np.zeros(MAX_WALKERS, dtype=np.float32)

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

@app.route("/run", methods=["POST"])
def run_simulation():
    try:
        data = request.json
        engine_choice = data.get("engine", "diffusion")
        walkers = int(data.get("walkers", 100000))
        steps = int(data.get("steps", 100))
        step_scale = float(data.get("step_scale", 1.0))
        user_code_string = data.get("code", "")

        pipeline.clear_rules()

        if user_code_string.strip():
            local_namespace = {}
            exec(user_code_string, {"np": np}, local_namespace)
            if "custom_rule" in local_namespace:
                pipeline.register_rule(local_namespace["custom_rule"])
            else:
                return jsonify({"status": "error", "message": "Your code must define a function named 'custom_rule(x, y)'."})

        active_x = global_x_buffer[:walkers]
        active_y = global_y_buffer[:walkers]
        active_x.fill(0.0)
        active_y.fill(0.0)

        selected_engine = ENGINES.get(engine_choice, engine_diffusion)
        engine_params = {"steps": steps, "step_scale": step_scale}
        
        final_x, final_y = pipeline.execute_pipeline(active_x, active_y, selected_engine, engine_params)

        return jsonify({"status": "success", "x": final_x.tolist(), "y": final_y.tolist()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
''')

print("Files generated cleanly!")
