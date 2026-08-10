import time
import numpy as np
from jit_compiler import auto_jit_rule

class SimulationPipeline:
    def __init__(self, max_walkers: int):
        self.max_walkers = max_walkers
        self.rules = []
        self.last_jit_success = True

    def register_rule(self, user_func):
        compiled_rule, success = auto_jit_rule(user_func)
        self.rules.append(compiled_rule)
        self.last_jit_success = success
        return success

    def clear_rules(self):
        self.rules = []
        self.last_jit_success = True

    def execute_pipeline(self, x_buffer, y_buffer, engine_runner_func, engine_params):
        start_total = time.perf_counter()
        
        start_engine = time.perf_counter()
        engine_runner_func(x_buffer, y_buffer, **engine_params)
        end_engine = time.perf_counter()
        engine_time = (end_engine - start_engine) * 1000.0

        x = np.asarray(x_buffer)
        y = np.asarray(y_buffer)
        
        for rule in self.rules:
            x, y = rule(x, y)
            
        end_total = time.perf_counter()
        total_time = (end_total - start_total) * 1000.0

        return x, y, engine_time, total_time
