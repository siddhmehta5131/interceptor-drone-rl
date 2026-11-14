"""Constant-acceleration Kalman tracker (plan P5.4, decision Q7, fix F4).

Per axis the state is x = [p, v, a]^T. The model is a constant-acceleration
model driven by white jerk (Bar-Shalom, Li & Kirubarajan, "Estimation with
Applications to Tracking and Navigation", 2001, section 6.2.3, the "white noise
jerk" model):

    x_{k+1} = F x_k + w_k,   w_k ~ N(0, Q)                      (EQ-KF-1)

    F = [[1, dt, dt^2/2],      Q = q_j * [[dt^5/20, dt^4/8, dt^3/6],
         [0,  1,  dt    ],                [dt^4/8,  dt^3/3, dt^2/2],
         [0,  0,  1     ]]                [dt^3/6,  dt^2/2, dt    ]]   (EQ-KF-2)

The seeker measures all three components (EQ-SEEK-1/2), so H = I and

    y_k = x_k + v_k,  v_k ~ N(0, R),  R = diag(sigma_p^2, sigma_v^2, sigma_a^2).  (EQ-KF-3)

R is the seeker noise at the *current* range, so it is derived from the noise
model rather than hand-set. The jerk spectral density is

    q_j = (a_T,max / tau_man)^2,                                  (EQ-KF-4)

i.e. the target can change its acceleration by its full limit over one
manoeuvre time scale ``tau_man`` (design parameter, default 1 s). On a dropout
only the prediction step runs, so the covariance grows and the filter coasts.

Update uses the Joseph form to stay symmetric positive definite:

    S = H P H^T + R,  K = P H^T S^-1,
    P+ = (I - K H) P (I - K H)^T + K R K^T.                       (EQ-KF-5)

The three axes share F, Q and R (isotropic noise), so one 3x3 covariance is
propagated and applied to all axes at once.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["KalmanTracker", "transition_matrices"]


def transition_matrices(dt: float, q_j: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (F, Q) of EQ-KF-1 / EQ-KF-2 for one axis."""
    F = np.array([[1.0, dt, 0.5 * dt ** 2],
                  [0.0, 1.0, dt],
                  [0.0, 0.0, 1.0]])
    Q = q_j * np.array([[dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
                        [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
                        [dt ** 3 / 6.0, dt ** 2 / 2.0, dt]])
    return F, Q


class KalmanTracker:
    """Constant-acceleration Kalman filter for a 3-D target."""

    def __init__(self, a_max: float, tau_man: float = 1.0):
        self.q_j = (float(a_max) / float(tau_man)) ** 2
        self.x = np.zeros((3, 3))      # rows: p, v, a ; columns: x, y, z
        self.P = np.eye(3)             # shared 3x3 covariance (per-axis)
        self.last_nis = float("nan")   # normalised innovation squared (per axis, mean)

    def reset(self, pos, vel, acc, sigma: Tuple[float, float, float]) -> None:
        self.x = np.vstack([np.asarray(pos, float), np.asarray(vel, float),
                            np.asarray(acc, float)])
        self.P = np.diag(np.square(sigma))
        self.last_nis = float("nan")

    def predict(self, dt: float) -> None:
        F, Q = transition_matrices(dt, self.q_j)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, pos, vel, acc, sigma: Tuple[float, float, float]) -> None:
        y = np.vstack([np.asarray(pos, float), np.asarray(vel, float),
                       np.asarray(acc, float)])
        R = np.diag(np.square(sigma))
        S = self.P + R                       # H = I
        K = self.P @ np.linalg.inv(S)
        innov = y - self.x
        # NIS averaged over the three Cartesian axes (each is chi^2 with 3 dof).
        Sinv = np.linalg.inv(S)
        self.last_nis = float(np.mean(np.einsum("ik,ij,jk->k", innov, Sinv, innov)))
        self.x = self.x + K @ innov
        I = np.eye(3)
        self.P = (I - K) @ self.P @ (I - K).T + K @ R @ K.T

    @property
    def pos(self) -> np.ndarray:
        return self.x[0].copy()

    @property
    def vel(self) -> np.ndarray:
        return self.x[1].copy()

    @property
    def acc(self) -> np.ndarray:
        return self.x[2].copy()
