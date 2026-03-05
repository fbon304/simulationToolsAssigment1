import math
import os
from time import perf_counter_ns

import assimulo.problem as apro
import assimulo.solvers as asol
import matplotlib.pyplot as plt
import numpy as np

# Parameters
k = 10
g = 1

ks = [10, 10000]
tol_values = [10**-8, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2]


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


def save_plots(xs, ys, out_dir: str):
    # 1) Coordinates vs time
    plt.figure()
    for i in range(ys.shape[0]):
        plt.plot(
            xs,
            ys[i],
            label=rf"Order {4 if i >= len(ks) else 2}, $k = {ks[i % len(ks)]}$",
        )
    plt.title(r"CVODE performance, RTOL = $10^-6$")
    plt.semilogx()
    plt.xlabel(r"ATOL")
    plt.ylabel(r"Simulation time")
    plt.legend()

    coord_path = os.path.join(out_dir, "perf_atol.png")
    plt.savefig(coord_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Output folder
    out_dir = "graphs_tols"  # relative path "./graphs"
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

    xs = tol_values
    ys = np.zeros((len(ks) * 2, len(tol_values)))

    i = 0
    for k_val in ks:
        k = k_val

        j = 0
        for tol in tol_values:
            atol = tol
            rtol = 10**-6

            # Define the problem in Assimulo (re-create per k for clarity)
            problem = apro.Explicit_Problem(spring_pendulum, t0=0, y0=y0)
            problem.name = f"Spring pendulum, $k={k}$"

            # --- Solver 1: BDF4 ---
            solver = asol.CVode(problem)
            solver.maxord = 4
            solver.atol = atol
            solver.rtol = rtol
            solver.reset()
            start = perf_counter_ns()
            t_sol, x_sol = solver.simulate(t_f, ncp)
            end = perf_counter_ns()
            ys[i, j] = end - start

            # --- Solver 2: BDF2 ---
            solver = asol.CVode(problem)
            solver.maxord = 2
            solver.atol = atol
            solver.rtol = rtol
            solver.reset()
            start = perf_counter_ns()
            t_sol, x_sol = solver.simulate(t_f, ncp)
            end = perf_counter_ns()
            ys[i + len(ks), j] = end - start

            j = j + 1
        i += 1

    save_plots(
        xs,
        ys,
        out_dir,
    )
    print(f"Saved all graphs to: {os.path.abspath(out_dir)}")
