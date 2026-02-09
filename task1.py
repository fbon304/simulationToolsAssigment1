import math

import assimulo.problem as apro
import assimulo.solvers as asol
import matplotlib.pyplot as plt
import numpy as np

# Parameters
k = 10**3
g = 1


# Define the first order ODE system
def lambda_fn(y1, y2):
    sqrt_term = math.sqrt(y1**2 + y2**2)
    return k * (sqrt_term - 1) / sqrt_term


def spring_pendulum(t, x):
    xdot = np.zeros_like(x)

    xdot[0] = x[2]
    xdot[1] = x[3]
    xdot[2] = -x[0] * lambda_fn(x[0], x[1])
    xdot[3] = -x[1] * lambda_fn(x[0], x[1]) - g

    return xdot


if __name__ == "__main__":
    # Simulation parameters
    t_f = 100
    ncp = 10000

    theta_0 = 90.0
    r_0 = 1.5

    # Define the problem in Assimulo
    problem = apro.Explicit_Problem(
        spring_pendulum,
        t0=0,
        y0=[
            r_0 * math.sin(theta_0 * math.pi / 180),
            r_0 * -math.cos(theta_0 * math.pi / 180),
            0,
            0,
        ],
    )
    problem.name = f"Spring pendulum, $k={k}$"

    # Initialize the solver
    solver = asol.CVode(problem)
    solver.reset()

    # Run the simulation
    t_sol, x_sol = solver.simulate(t_f, ncp)

    # Plot
    plt.figure()
    plt.plot(t_sol, x_sol[:, 0])
    plt.plot(t_sol, x_sol[:, 1])
    plt.title(problem.name)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x_1, x_2(t)$")
    plt.show()

    plt.figure()
    plt.plot(x_sol[:, 0], x_sol[:, 1])
    plt.title(problem.name)
    plt.xlabel(r"$x_1(t)$")
    plt.ylabel(r"$x_2(t)$")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.show()
