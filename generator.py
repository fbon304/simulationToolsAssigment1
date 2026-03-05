import math
import os

import assimulo.problem as apro
import assimulo.solvers as asol
import matplotlib.pyplot as plt
import numpy as np

from BDF4_Assimulo import BDF_4_Newton
from BDF2_Assimulo import BDF_2

# Parameters
k = 1  # will be overwritten inside the loop
g = 1

k_values = [10, 100, 1000, 10000, 100000]

# Define the first order ODE system
def lambda_fn(y1, y2):
    sqrt_term = math.hypot(y1, y2)
    return k * (sqrt_term - 1) / sqrt_term

def spring_pendulum(t, x):
    xdot = np.zeros_like(x)

    xdot[0] = x[2]
    xdot[1] = x[3]
    xdot[2] = -x[0] * lambda_fn(x[0], x[1])
    xdot[3] = -x[1] * lambda_fn(x[0], x[1]) - g

    return xdot

def save_plots(t_sol, x_sol, method_name: str, h_value: str, k_value: int, out_dir: str):
    # 1) Coordinates vs time
    plt.figure()
    plt.plot(t_sol, x_sol[:, 0], label=r"$x_1(t)$")
    plt.plot(t_sol, x_sol[:, 1], label=r"$x_2(t)$")
    plt.title(f"Spring pendulum, $k={k_value}$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x_1, x_2(t)$")
    plt.legend()

    coord_path = os.path.join(out_dir, f"Coordinates_{method_name}_{h_value}_k{k_value}.png")
    plt.savefig(coord_path, dpi=300)
    plt.close()

    # 2) Phase / trajectory in x1-x2
    plt.figure()
    plt.plot(x_sol[:, 0], x_sol[:, 1])
    plt.title(f"Spring pendulum, $k={k_value}$")
    plt.xlabel(r"$x_1(t)$")
    plt.ylabel(r"$x_2(t)$")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    phase_path = os.path.join(out_dir, f"Phase_{method_name}_{h_value}_k{k_value}.png")
    plt.savefig(phase_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Output folder
    out_dir = "graphs"  # relative path "./graphs"
    os.makedirs(out_dir, exist_ok=True)

    # Simulation parameters
    t_f = 100
    ncp = 10000

    # Initial condition
    theta_0 = 90.0
    r_0 = 1.1

    y0 = [
        r_0 * math.sin(theta_0 * math.pi / 180),
        r_0 * -math.cos(theta_0 * math.pi / 180),
        0.0,
        0.0,
    ]

    # Step sizes (set what you want here)
    h_bdf = 0.001
    h_euler = 0.001

    for k_val in k_values:
        # update global k used by lambda_fn
        k = k_val

        # Define the problem in Assimulo (re-create per k for clarity)
        problem = apro.Explicit_Problem(spring_pendulum, t0=0, y0=y0)
        problem.name = f"Spring pendulum, $k={k}$"

        # --- Solver 1: BDF4 ---
        solver = BDF_4_Newton(problem)
        solver.reset()
        t_sol, x_sol = solver.simulate(t_f, ncp)
        save_plots(
            t_sol, x_sol,
            method_name="BDF4",
            h_value=f"h{h_bdf}",
            k_value=k,
            out_dir=out_dir
        )

        # --- Solver 2: BDF2 ---
        solver = BDF_2(problem)
        solver.reset()
        t_sol, x_sol = solver.simulate(t_f, ncp)
        save_plots(
            t_sol, x_sol,
            method_name="BDF2",
            h_value=f"h{h_bdf}",
            k_value=k,
            out_dir=out_dir
        )

        # --- Solver 3: Explicit Euler (Assimulo) ---
        solver = asol.ExplicitEuler(problem)
        solver.h = h_euler
        t_sol, x_sol = solver.simulate(t_f, ncp)
        save_plots(
            t_sol, x_sol,
            method_name="Euler",
            h_value=f"h{h_euler}",
            k_value=k,
            out_dir=out_dir
        )

    print(f"Saved all graphs to: {os.path.abspath(out_dir)}")