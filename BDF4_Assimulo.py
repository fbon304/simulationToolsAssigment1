from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import scipy.linalg as SL


class BDF_4_Newton(Explicit_ODE):
    """
    BDF-4 with Newton iteration.
    """

    tol = 1e-10
    maxit = 30
    maxsteps = 50000
    fd_eps = 1e-8  # finite-difference step for numerical Jacobian

    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)

        self.options["h"] = 0.01

        self.statistics["nsteps"] = 0
        self.statistics["nfcns"]  = 0
        self.nnewton = 0

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
        """
        Integrates (t, y) with fixed step until t >= tf.
        Startup is done with implicit BDF1/BDF2/BDF3, then BDF4.
        """
        h = self.options["h"]
        h = min(h, abs(tf - t))

        tres = []
        yres = []

        t = float(t)
        y = np.asarray(y, dtype=float)
        # Initialize history so tuple-shift works from the first step onward.
        # Real history is built progressively by BDF1 -> BDF2 -> BDF3 startup.
        t_nm1 = t
        t_nm2 = t
        t_nm3 = t
        y_nm1 = y.copy()
        y_nm2 = y.copy()
        y_nm3 = y.copy()

        for i in range(self.maxsteps):
            if t >= tf:
                break

            self.statistics["nsteps"] += 1
            h = min(self.h, abs(tf - t))

            if i == 0:
                t_np1, y_np1 = self.step_BDF1_Newton(t, y, h)
            elif i == 1:
                t_np1, y_np1 = self.step_BDF2_Newton([t, t_nm1], [y, y_nm1], h)
            elif i == 2:
                t_np1, y_np1 = self.step_BDF3_Newton([t, t_nm1, t_nm2], [y, y_nm1, y_nm2], h)
            else:
                T = [t, t_nm1, t_nm2, t_nm3]
                Y = [y, y_nm1, y_nm2, y_nm3]
                t_np1, y_np1 = self.step_BDF4_Newton(T, Y, h)

            t, t_nm1, t_nm2, t_nm3 = t_np1, t, t_nm1, t_nm2
            y, y_nm1, y_nm2, y_nm3 = y_np1, y, y_nm1, y_nm2

            tres.append(t)
            yres.append(y.copy())
        else:
            raise ODE_Exception("Final time not reached within maximum number of steps")

        return ID_PY_OK, tres, yres

    def _step_BDF_Newton(self, alpha, t_n, y_vals, h):
        """
        Generic implicit BDF step solved with full Newton.
        alpha = [alpha0, alpha1, ..., alphak]
        y_vals = [y_n, y_n-1, ..., y_n-(k-1)]
        """
        ys = [np.asarray(v, dtype=float) for v in y_vals]
        t_np1 = t_n + h
        yk = ys[0].copy()
        I = np.eye(yk.size)

        const = np.zeros_like(yk)
        for j in range(1, len(alpha)):
            const += alpha[j] * ys[j - 1]

        for _ in range(self.maxit):
            self.nnewton += 1

            fk = self._rhs(t_np1, yk)
            G = alpha[0]*yk + const - h*fk

            if SL.norm(G) < self.tol:
                return t_np1, yk

            try:
                Jf = self._jac(t_np1, yk)
                JG = alpha[0]*I - h*Jf
                delta = SL.solve(JG, -G, assume_a='gen')
            except Exception as e:
                raise ODE_Exception(f"Newton failed at t={t_np1}: {e}")

            y_next = yk + delta
            if SL.norm(delta) < self.tol * (1.0 + SL.norm(y_next)):
                return t_np1, y_next
            yk = y_next

        raise ODE_Exception(f"Newton did not converge within {self.maxit} iterations at t={t_np1}")

    def step_BDF1_Newton(self, t_n, y_n, h):
        alpha = [1.0, -1.0]
        return self._step_BDF_Newton(alpha, t_n, [y_n], h)

    def step_BDF2_Newton(self, T, Y, h):
        alpha = [3.0/2.0, -2.0, 1.0/2.0]
        t_n, _ = T
        y_n, y_nm1 = Y
        return self._step_BDF_Newton(alpha, t_n, [y_n, y_nm1], h)

    def step_BDF3_Newton(self, T, Y, h):
        alpha = [11.0/6.0, -3.0, 3.0/2.0, -1.0/3.0]
        t_n, _, _ = T
        y_n, y_nm1, y_nm2 = Y
        return self._step_BDF_Newton(alpha, t_n, [y_n, y_nm1, y_nm2], h)

    def step_BDF4_Newton(self, T, Y, h):
        """
        BDF-4 with Newton iteration.

        alpha0*y_{n+1}+alpha1*y_n+alpha2*y_{n-1}+alpha3*y_{n-2}+alpha4*y_{n-3} = h f(t_{n+1}, y_{n+1})
        alpha = [25/12, -4, 3, -4/3, 1/4]
        """
        t_n = T[0]
        y_n, y_nm1, y_nm2, y_nm3 = Y
        alpha = [25.0/12.0, -4.0, 3.0, -4.0/3.0, 1.0/4.0]
        return self._step_BDF_Newton(alpha, t_n, [y_n, y_nm1, y_nm2, y_nm3], h)

    def print_statistics(self, verbose=NORMAL):
        self.log_message(f'Final Run Statistics            : {self.problem.name} \n', verbose)
        self.log_message(f' Step-length                    : {self.options["h"]} ', verbose)
        self.log_message(' Number of Steps                : ' + str(self.statistics["nsteps"]), verbose)
        self.log_message(' Number of Function Evaluations : ' + str(self.statistics["nfcns"]), verbose)
        self.log_message(' Newton iterations              : ' + str(self.nnewton), verbose)

        self.log_message('\nSolver options:\n', verbose)
        self.log_message(' Solver            : BDF4', verbose)
        self.log_message(' Solver type       : Newton iteration\n', verbose)


# if __name__ == "__main__":
#     # ----------------- Example usage (optional) -----------------
#     def pend(t, y):
#         gl = 13.7503671
#         return np.array([y[1], -gl*np.sin(y[0])])
#
#     pend_mod = Explicit_Problem(pend, y0=np.array([2.*np.pi, 1.]))
#     pend_mod.name = 'Nonlinear Pendulum (BDF4 Newton)'
#
#     sim = BDF_4_Newton(pend_mod)
#     t, y = sim.simulate(1)
#     sim.plot()
#     mpl.show()
    
