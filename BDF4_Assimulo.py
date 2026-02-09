from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL

class BDF_4_Newton(Explicit_ODE):
    """
    BDF-4 with Newton iteration.
    """

    tol = 1e-10
    maxit = 30
    maxsteps = 500
    fd_eps = 1e-8  # finite-difference step for numerical Jacobian

    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)

        self.options["h"] = 0.01

        self.statistics["nsteps"] = 0
        self.statistics["nfcns"]  = 0
        self.statistics["nnewton"] = 0

        self._need_new_jac = True
        self._J_cached = None
        self._J_cache_key = None  # (t_np1, h) to detect changes

    def _set_h(self, h):
        self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]

    h = property(_get_h, _set_h)

    # ---------- helpers ----------
    def _rhs(self, t, y):
        self.statistics["nfcns"] += 1
        return self.problem.rhs(t, y)

    def _jac(self, t, y):
        y = np.asarray(y, dtype=float)
        n = y.size
        J = np.zeros((n, n), dtype=float)

        f0 = self._rhs(t, y)  # counts as 1 f eval
        eps = self.fd_eps

        for j in range(n):
            ypert = y.copy()
            dy = eps * (1.0 + abs(y[j]))
            ypert[j] += dy
            fj = self._rhs(t, ypert)  # counts as 1 f eval
            J[:, j] = (fj - f0) / dy

        return J

    # ---------- integration ----------
    def integrate(self, t, y, tf, opts):
        h = min(self.options["h"], abs(tf - t))

        tres, yres = [], []

        # store last values (we need y_n, y_{n-1}, y_{n-2}, y_{n-3})
        t_hist = [t]
        y_hist = [np.array(y, dtype=float)]

        for i in range(self.maxsteps):
            if t >= tf:
                break

            self.statistics["nsteps"] += 1
            h = min(self.h, abs(tf - t))

            # Startup (simple): first 3 steps with Explicit Euler
            if len(y_hist) < 4:
                t_np1, y_np1 = self.step_EE(t, y, h)
            else:
                T = [t_hist[-1], t_hist[-2], t_hist[-3], t_hist[-4]]
                Y = [y_hist[-1], y_hist[-2], y_hist[-3], y_hist[-4]]

                # ---- if Newton fails, force new Jacobian next time ----
                try:
                    t_np1, y_np1 = self.step_BDF4_Newton(T, Y, h)
                except Explicit_ODE_Exception:
                    self._need_new_jac = True
                    self._J_cached = None
                    self._J_cache_key = None
                    raise

            t, y = t_np1, y_np1

            t_hist.append(t)
            y_hist.append(y.copy())
            # keep only last 4
            if len(y_hist) > 4:
                t_hist = t_hist[-4:]
                y_hist = y_hist[-4:]

            tres.append(t)
            yres.append(y.copy())
        else:
            raise Explicit_ODE_Exception("Final time not reached within maximum number of steps")

        return ID_PY_OK, tres, yres

    def step_EE(self, t, y, h):
        f = self.problem.rhs
        self.statistics["nfcns"] += 1
        return t + h, np.asarray(y, dtype=float) + h * f(t, y)

    # ---------- BDF4 Newton step ----------
    def step_BDF4_Newton(self, T, Y, h):
        """
        BDF-4 with Newton iteration (simplified Newton with Jacobian reuse).

        alpha0*y_{n+1}+alpha1*y_n+alpha2*y_{n-1}+alpha3*y_{n-2}+alpha4*y_{n-3} = h f(t_{n+1}, y_{n+1})
        alpha = [25/12, -4, 3, -4/3, 1/4]
        """
        alpha = [25.0/12.0, -4.0, 3.0, -4.0/3.0, 1.0/4.0]

        t_n, t_nm1, t_nm2, t_nm3 = T
        y_n, y_nm1, y_nm2, y_nm3 = [np.asarray(v, dtype=float) for v in Y]

        t_np1 = t_n + h

        # predictor (simple): constant extrapolation
        yk = y_n.copy()

        # constant part
        const = alpha[1]*y_n + alpha[2]*y_nm1 + alpha[3]*y_nm2 + alpha[4]*y_nm3

        I = np.eye(y_n.size)

        # ---- added: decide if we need a new Jacobian (minimal) ----
        key = (float(t_np1), float(h))
        if self._need_new_jac or (self._J_cache_key != key) or (self._J_cached is None):
            # Build J_G = alpha0*I - h*J_f at the predictor point yk
            Jf = self._jac(t_np1, yk)
            self._J_cached = alpha[0]*I - h*Jf
            self._J_cache_key = key
            self._need_new_jac = False

        # Newton iterations, reusing the cached J
        for it in range(self.maxit):
            self.statistics["nnewton"] += 1

            fk = self._rhs(t_np1, yk)
            G  = alpha[0]*yk + const - h*fk

            if SL.norm(G) < self.tol:
                return t_np1, yk

            # ---- changed: reuse cached Jacobian, no recomputation here ----
            try:
                delta = SL.solve(self._J_cached, -G, assume_a='gen')
            except Exception as e:
                # If linear solve fails, force recompute next time
                self._need_new_jac = True
                raise Explicit_ODE_Exception(f"Linear solve failed: {e}")

            y_next = yk + delta

            if SL.norm(delta) < self.tol * (1.0 + SL.norm(y_next)):
                return t_np1, y_next

            yk = y_next

        # If Newton didn't converge, force new Jacobian next step/attempt
        self._need_new_jac = True
        raise Explicit_ODE_Exception(f"Newton did not converge within {self.maxit} iterations")

    def print_statistics(self, verbose=NORMAL):
        self.log_message(f'Final Run Statistics            : {self.problem.name} \n', verbose)
        self.log_message(f' Step-length                    : {self.options["h"]} ', verbose)
        self.log_message(' Number of Steps                : ' + str(self.statistics["nsteps"]), verbose)
        self.log_message(' Number of Function Evaluations : ' + str(self.statistics["nfcns"]), verbose)
        self.log_message(' Newton iterations              : ' + str(self.statistics["nnewton"]), verbose)
        self.log_message('\nSolver options:\n', verbose)
        self.log_message(' Solver            : BDF4 (Newton, Jacobian reuse)', verbose)
        self.log_message(' Solver type       : Fixed step\n', verbose)


# ----------------- Example usage (same as yours) -----------------
def pend(t, y):
    gl = 13.7503671
    return np.array([y[1], -gl*np.sin(y[0])])

pend_mod = Explicit_Problem(pend, y0=np.array([2.*np.pi, 1.]))
pend_mod.name = 'Nonlinear Pendulum (BDF4 Newton)'

sim = BDF_4_Newton(pend_mod)
t, y = sim.simulate(1)
sim.plot()
mpl.show()
