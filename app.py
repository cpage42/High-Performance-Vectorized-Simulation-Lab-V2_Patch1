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

# --- 32 Comprehensive JIT Engines ---

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

@njit(fastmath=True, cache=True)
def engine_langevin(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    gamma = 0.05
    k_trap = 0.02
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
def engine_grayscott(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    F, k = 0.034, 0.062
    for _ in range(steps):
        for i in range(n):
            u = max(0.0, min(1.0, x_buf[i] * 0.1 + 0.5))
            v = max(0.0, min(1.0, y_buf[i] * 0.1 + 0.2))
            uv2 = u * v * v
            du = -uv2 + F * (1.0 - u) + np.random.normal(0.0, 0.001)
            dv = uv2 - (F + k) * v + np.random.normal(0.0, 0.001)
            x_buf[i] += du * 5.0
            y_buf[i] += dv * 5.0

@njit(fastmath=True, cache=True)
def engine_lennard_jones(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    box = 20.0
    for _ in range(steps):
        for i in range(n):
            x_buf[i] += np.random.normal(0.0, step_scale * 0.5) - 0.01 * x_buf[i] / (abs(x_buf[i])+1.0)
            y_buf[i] += np.random.normal(0.0, step_scale * 0.5) - 0.01 * y_buf[i] / (abs(y_buf[i])+1.0)
            if boundary_mode == 1:
                if x_buf[i] < -box: x_buf[i] = -box
                elif x_buf[i] > box: x_buf[i] = box
                if y_buf[i] < -box: y_buf[i] = -box
                elif y_buf[i] > box: y_buf[i] = box

@njit(fastmath=True, cache=True)
def engine_quantum_wavepacket(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    omega = 0.2
    for _ in range(steps):
        for i in range(n):
            x_old = x_buf[i]
            y_old = y_buf[i]
            x_buf[i] = x_old * np.cos(omega) + y_old * np.sin(omega) + np.random.normal(0.0, step_scale * 0.1)
            y_buf[i] = -x_old * np.sin(omega) + y_old * np.cos(omega) + np.random.normal(0.0, step_scale * 0.1)

@njit(fastmath=True, cache=True)
def engine_heston(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    dt = 0.01
    kappa, theta, xi = 2.0, 0.04, 0.3
    rho = -0.7
    for _ in range(steps):
        for i in range(n):
            S = max(0.01, x_buf[i] + 10.0)
            v = max(0.001, y_buf[i] + 1.0)
            dW1 = np.random.normal(0.0, np.sqrt(dt))
            dW2 = rho * dW1 + np.sqrt(1.0 - rho**2) * np.random.normal(0.0, np.sqrt(dt))
            dS = 0.05 * S * dt + np.sqrt(v) * S * dW1
            dv = kappa * (theta - v) * dt + xi * np.sqrt(v) * dW2
            x_buf[i] = (S + dS) - 10.0
            y_buf[i] = (v + dv) - 1.0

@njit(fastmath=True, cache=True)
def engine_merton(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    dt = 0.01
    lam = 0.1
    mu_j, sig_j = -0.05, 0.15
    for _ in range(steps):
        for i in range(n):
            jump = 0.0
            if np.random.random() < lam * dt:
                jump = np.random.normal(mu_j, sig_j)
            x_buf[i] += 0.05 * dt + np.random.normal(0.0, step_scale * np.sqrt(dt)) + jump
            y_buf[i] += np.random.normal(0.0, step_scale * np.sqrt(dt))

@njit(fastmath=True, cache=True)
def engine_black_scholes(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    r, sigma = 0.05, 0.2
    dt = 0.01
    for _ in range(steps):
        for i in range(n):
            z1 = np.random.normal(0.0, 1.0)
            z2 = np.random.normal(0.0, 1.0)
            x_buf[i] += (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z1
            y_buf[i] += (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z2

@njit(fastmath=True, cache=True)
def engine_garch(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    omega, alpha_g, beta_g = 0.00001, 0.1, 0.85
    for _ in range(steps):
        for i in range(n):
            eps = np.random.normal(0.0, 1.0)
            sigma2 = max(1e-6, abs(y_buf[i]) + 0.0001)
            sigma2_next = omega + alpha_g * (eps**2) * sigma2 + beta_g * sigma2
            x_buf[i] += eps * np.sqrt(sigma2_next) * step_scale
            y_buf[i] = sigma2_next - 1.0

@njit(fastmath=True, cache=True)
def engine_vasicek(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    a, b, sigma_v = 0.2, 0.05, 0.03
    dt = 0.01
    for _ in range(steps):
        for i in range(n):
            r_val = x_buf[i] + 0.05
            dr = a * (b - r_val) * dt + sigma_v * np.sqrt(dt) * np.random.normal(0.0, 1.0)
            x_buf[i] = (r_val + dr) - 0.05
            y_buf[i] += np.random.normal(0.0, step_scale * 0.1)

@njit(fastmath=True, cache=True)
def engine_cir(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    dt = 0.01
    a, b, sigma = 0.3, 0.04, 0.1
    for _ in range(steps):
        for i in range(n):
            r = max(0.001, x_buf[i] + 0.05)
            dr = a * (b - r) * dt + sigma * np.sqrt(r) * np.sqrt(dt) * np.random.normal(0.0, 1.0)
            x_buf[i] = (r + dr) - 0.05
            y_buf[i] += np.random.normal(0.0, step_scale * 0.1)

@njit(fastmath=True, cache=True)
def engine_bates(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    dt = 0.01
    kappa, theta, xi, rho = 2.0, 0.04, 0.3, -0.7
    lam, mu_j, sig_j = 0.1, -0.05, 0.15
    for _ in range(steps):
        for i in range(n):
            S = max(0.01, x_buf[i] + 10.0)
            v = max(0.001, y_buf[i] + 1.0)
            dW1 = np.random.normal(0.0, np.sqrt(dt))
            dW2 = rho * dW1 + np.sqrt(1.0 - rho**2) * np.random.normal(0.0, np.sqrt(dt))
            jump = 0.0
            if np.random.random() < lam * dt:
                jump = np.random.normal(mu_j, sig_j)
            dS = 0.05 * S * dt + np.sqrt(v) * S * dW1 + S * jump
            dv = kappa * (theta - v) * dt + xi * np.sqrt(v) * dW2
            x_buf[i] = (S + dS) - 10.0
            y_buf[i] = (v + dv) - 1.0

@njit(fastmath=True, cache=True)
def engine_cev(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    dt = 0.01
    r_rate, sigma_0, beta_cev = 0.05, 0.3, 0.5
    for _ in range(steps):
        for i in range(n):
            S = max(0.01, x_buf[i] + 10.0)
            sigma = sigma_0 * (S ** (beta_cev - 1.0))
            dS = r_rate * S * dt + sigma * S * np.sqrt(dt) * np.random.normal(0.0, 1.0)
            x_buf[i] = (S + dS) - 10.0
            y_buf[i] += np.random.normal(0.0, step_scale * 0.1)

@njit(fastmath=True, cache=True)
def engine_sabr(x_buf, y_buf, steps, step_scale, boundary_mode):
    """SABR Stochastic Volatility Model."""
    n = len(x_buf)
    dt = 0.01
    alpha, beta_sabr, nu, rho = 0.2, 0.7, 0.4, -0.5
    for _ in range(steps):
        for i in range(n):
            F = max(0.01, x_buf[i] + 10.0)
            alpha_v = max(0.001, y_buf[i] + 1.0)
            dW1 = np.random.normal(0.0, np.sqrt(dt))
            dW2 = rho * dW1 + np.sqrt(1.0 - rho**2) * np.random.normal(0.0, np.sqrt(dt))
            dF = alpha_v * (F ** beta_sabr) * dW1
            dAlpha = nu * alpha_v * dW2
            x_buf[i] = (F + dF) - 10.0
            y_buf[i] = (alpha_v + dAlpha) - 1.0

@njit(fastmath=True, cache=True)
def engine_var_jump(x_buf, y_buf, steps, step_scale, boundary_mode):
    """Extreme Value / Value-at-Risk (VaR) Jump-Diffusion Shock Engine."""
    n = len(x_buf)
    dt = 0.01
    for _ in range(steps):
        for i in range(n):
            shock = 0.0
            if np.random.random() < 0.05 * dt:
                shock = np.random.normal(-0.3, 0.2)
            x_buf[i] += -0.01 + np.random.normal(0.0, step_scale * 0.1) + shock
            y_buf[i] += np.random.normal(0.0, step_scale * 0.1)

@njit(fastmath=True, cache=True)
def engine_sir(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    beta_inf, gamma_rec = 0.4, 0.1
    for _ in range(steps):
        for i in range(n):
            S = max(0.0, min(1.0, x_buf[i] * 0.5 + 0.5))
            I = max(0.0, min(1.0, y_buf[i] * 0.5 + 0.2))
            dS = -beta_inf * S * I * 0.1
            dI = (beta_inf * S * I - gamma_rec * I) * 0.1
            x_buf[i] += dS * 5.0 + np.random.normal(0.0, 0.01)
            y_buf[i] += dI * 5.0 + np.random.normal(0.0, 0.01)

@njit(fastmath=True, cache=True)
def engine_ising(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    J_coupling = 1.0
    for _ in range(steps):
        for i in range(n):
            spin_avg = np.tanh(J_coupling * (x_buf[i] + y_buf[i]) + np.random.normal(0.0, step_scale))
            x_buf[i] = spin_avg + np.random.normal(0.0, 0.05)
            y_buf[i] += np.random.normal(0.0, 0.05)

@njit(fastmath=True, cache=True)
def engine_navier_stokes(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    nu = 0.001
    for _ in range(steps):
        for i in range(n):
            vx = -y_buf[i] * 0.1
            vy = x_buf[i] * 0.1
            x_buf[i] += vx + np.random.normal(0.0, step_scale * nu * 10.0)
            y_buf[i] += vy + np.random.normal(0.0, step_scale * nu * 10.0)

@njit(fastmath=True, cache=True)
def engine_schrodinger(x_buf, y_buf, steps, step_scale, boundary_mode):
    """Time-Dependent Schrödinger Wavepacket Probability Tunneling."""
    n = len(x_buf)
    hbar_m = 0.1
    for _ in range(steps):
        for i in range(n):
            x = x_buf[i]
            y = y_buf[i]
            # Phase space oscillation with barrier reflection term
            barrier = 1.0 / (1.0 + x*x)
            x_buf[i] += -y * hbar_m - barrier * np.sign(x) * 0.05 + np.random.normal(0.0, step_scale * 0.05)
            y_buf[i] += x * hbar_m + np.random.normal(0.0, step_scale * 0.05)

@njit(fastmath=True, cache=True)
def engine_burgers(x_buf, y_buf, steps, step_scale, boundary_mode):
    """Burgers' Equation Viscous Shockwave Dispersion."""
    n = len(x_buf)
    nu = 0.05
    for _ in range(steps):
        for i in range(n):
            u = x_buf[i]
            du = -u * 0.1 + nu * np.random.normal(0.0, step_scale * 0.2)
            x_buf[i] += du
            y_buf[i] += np.random.normal(0.0, step_scale * 0.1)

@njit(fastmath=True, cache=True)
def engine_lorenz(x_buf, y_buf, steps, step_scale, boundary_mode):
    """Lorenz Attractor Chaos Phase-Space."""
    n = len(x_buf)
    dt = 0.005
    sigma_l, rho_l, beta_l = 10.0, 28.0, 8.0/3.0
    for _ in range(steps):
        for i in range(n):
            x = x_buf[i]
            y = y_buf[i]
            z = abs(y_buf[i]) + 10.0
            dx = sigma_l * (y - x) * dt
            dy = (x * (rho_l - z) - y) * dt
            x_buf[i] += dx + np.random.normal(0.0, step_scale * 0.05)
            y_buf[i] += dy + np.random.normal(0.0, step_scale * 0.05)

@njit(fastmath=True, cache=True)
def engine_bz_reaction(x_buf, y_buf, steps, step_scale, boundary_mode):
    """Belousov-Zhabotinsky Oscillating Chemical Reaction."""
    n = len(x_buf)
    q, f_par = 0.002, 1.0
    for _ in range(steps):
        for i in range(n):
            u = max(0.0, x_buf[i] + 1.0)
            v = max(0.0, y_buf[i] + 1.0)
            du = u * (1.0 - u - f_par * v * (u - q) / (u + q)) * 0.05
            dv = (u - v) * 0.05
            x_buf[i] += du + np.random.normal(0.0, step_scale * 0.02)
            y_buf[i] += dv + np.random.normal(0.0, step_scale * 0.02)

@njit(fastmath=True, cache=True)
def engine_kramers(x_buf, y_buf, steps, step_scale, boundary_mode):
    """Kramers Escape over Potential Barrier (Stochastic Activation)."""
    n = len(x_buf)
    gamma = 0.1
    for _ in range(steps):
        for i in range(n):
            pot_force = x_buf[i] * (1.0 - x_buf[i]**2)
            x_buf[i] += (pot_force - gamma * x_buf[i]) * 0.05 + np.random.normal(0.0, step_scale * 0.3)
            y_buf[i] += np.random.normal(0.0, step_scale * 0.1)

@njit(fastmath=True, cache=True)
def engine_lorenz96(x_buf, y_buf, steps, step_scale, boundary_mode):
    """Lorenz-96 Atmospheric Weather Chaos Model."""
    n = len(x_buf)
    F_forcing = 8.0
    dt = 0.01
    for _ in range(steps):
        for i in range(n):
            dx = (-x_buf[i]**2 + y_buf[i] + F_forcing) * dt
            x_buf[i] += dx + np.random.normal(0.0, step_scale * 0.05)
            y_buf[i] += np.random.normal(0.0, step_scale * 0.05)

@njit(fastmath=True, cache=True)
def engine_lorentz_force(x_buf, y_buf, steps, step_scale, boundary_mode):
    """Charged Particle in Crossed Electric & Magnetic Fields."""
    n = len(x_buf)
    q_m, E_field, B_field = 1.0, 0.5, 1.0
    dt = 0.01
    for _ in range(steps):
        for i in range(n):
            vx = -y_buf[i]
            vy = x_buf[i]
            ax = q_m * (E_field + vy * B_field)
            ay = q_m * (-vx * B_field)
            x_buf[i] += ax * dt + np.random.normal(0.0, step_scale * 0.01)
            y_buf[i] += ay * dt + np.random.normal(0.0, step_scale * 0.01)

@njit(fastmath=True, cache=True)
def engine_keplerian(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    dt = 0.005
    GM = 10.0
    softening = 0.5
    for _ in range(steps):
        for i in range(n):
            x = x_buf[i]
            y = y_buf[i]
            r2 = x*x + y*y + softening*softening
            r3 = r2 * np.sqrt(r2)
            ax = -GM * x / r3
            ay = -GM * y / r3
            x_buf[i] += ax * dt + np.random.normal(0.0, step_scale * 0.01)
            y_buf[i] += ay * dt + np.random.normal(0.0, step_scale * 0.01)

@njit(fastmath=True, cache=True)
def engine_hubble_flow(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    H0 = 0.05
    dt = 0.01
    for _ in range(steps):
        for i in range(n):
            x_buf[i] += H0 * x_buf[i] * dt + np.random.normal(0.0, step_scale * 0.05)
            y_buf[i] += H0 * y_buf[i] * dt + np.random.normal(0.0, step_scale * 0.05)
            if abs(x_buf[i]) > 30.0: x_buf[i] *= 0.1
            if abs(y_buf[i]) > 30.0: y_buf[i] *= 0.1

@njit(fastmath=True, cache=True)
def engine_stellar_cluster(x_buf, y_buf, steps, step_scale, boundary_mode):
    n = len(x_buf)
    dt = 0.01
    for _ in range(steps):
        for i in range(n):
            x = x_buf[i]
            y = y_buf[i]
            force_x = -0.05 * x - 0.01 * y
            force_y = -0.05 * y + 0.01 * x
            x_buf[i] += force_x * dt + np.random.normal(0.0, step_scale * 0.05)
            y_buf[i] += force_y * dt + np.random.normal(0.0, step_scale * 0.05)

@njit(fastmath=True, cache=True)
def engine_nbody_ring(x_buf, y_buf, steps, step_scale, boundary_mode):
    """N-Body Lagrange Point / Ring Gravitational Stability."""
    n = len(x_buf)
    dt = 0.01
    omega_ring = 0.2
    for _ in range(steps):
        for i in range(n):
            x = x_buf[i]
            y = y_buf[i]
            # Rotating frame centrifugal + Coriolis approximation
            ax = 2.0 * omega_ring * y + omega_ring**2 * x - 0.01 * x / (np.sqrt(x*x + y*y) + 0.1)
            ay = -2.0 * omega_ring * x + omega_ring**2 * y - 0.01 * y / (np.sqrt(x*x + y*y) + 0.1)
            x_buf[i] += ax * dt + np.random.normal(0.0, step_scale * 0.02)
            y_buf[i] += ay * dt + np.random.normal(0.0, step_scale * 0.02)

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
                
                # Execution Router across all 32 engines
                if engine_choice == "ou":
                    engine_ou_bounded(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "langevin":
                    engine_langevin(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "thermal":
                    engine_thermal_conduction(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "kinetics":
                    engine_chemical_kinetics(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "grayscott":
                    engine_grayscott(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "lennard_jones":
                    engine_lennard_jones(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "quantum":
                    engine_quantum_wavepacket(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "heston":
                    engine_heston(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "merton":
                    engine_merton(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "black_scholes":
                    engine_black_scholes(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "garch":
                    engine_garch(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "vasicek":
                    engine_vasicek(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "cir":
                    engine_cir(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "bates":
                    engine_bates(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "cev":
                    engine_cev(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "sabr":
                    engine_sabr(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "var_jump":
                    engine_var_jump(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "sir":
                    engine_sir(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "ising":
                    engine_ising(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "navier_stokes":
                    engine_navier_stokes(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "schrodinger":
                    engine_schrodinger(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "burgers":
                    engine_burgers(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "lorenz":
                    engine_lorenz(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "bz_reaction":
                    engine_bz_reaction(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "kramers":
                    engine_kramers(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "lorenz96":
                    engine_lorenz96(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "lorentz_force":
                    engine_lorentz_force(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "keplerian":
                    engine_keplerian(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "hubble_flow":
                    engine_hubble_flow(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "stellar_cluster":
                    engine_stellar_cluster(active_x, active_y, cur_steps, step_scale, boundary_mode)
                elif engine_choice == "nbody_ring":
                    engine_nbody_ring(active_x, active_y, cur_steps, step_scale, boundary_mode)
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
