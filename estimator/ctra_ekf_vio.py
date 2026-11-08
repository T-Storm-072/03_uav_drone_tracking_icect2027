"""
3D Constant-Turn-Rate and Acceleration Extended Kalman Filter (CTRA-EKF)
with Tightly Coupled Monocular VIO Ego-Motion Compensation.
Authors: Javeria Dawood, Syed Ali Turab Kazmi, Amber Saeed
Department of Robotics & Intelligent Systems, Bahria University, Islamabad
"""

import math
import numpy as np


class CTRAExtendedKalmanFilter3D:
    """
    8-state 3D CTRA Extended Kalman Filter for micro-UAV trajectory tracking
    and short-term prediction across 1.0-3.0 s full visual occlusions.
    State vector x = [x, y, z, v_xy, psi, omega_psi, a_xy, v_z]^T
    """

    def __init__(self, dt: float = 1.0 / 30.0):
        self.dt = dt
        self.x = np.zeros(8, dtype=float)
        self.P = np.eye(8, dtype=float) * 0.5

        # Process noise covariance Q
        self.Q = np.diag([
            0.004, 0.004, 0.004,  # x, y, z (m^2)
            0.025,                # v_xy (m/s)^2
            0.008,                # psi (rad^2)
            0.015,                # omega_psi (rad/s)^2
            0.040,                # a_xy (m/s^2)^2
            0.015,                # v_z (m/s)^2
        ])

        # Measurement noise covariance R for [x, y, z] back-projected position
        self.R = np.diag([0.035, 0.035, 0.045])
        self.H = np.zeros((3, 8), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

    def initialize(self, pos_xyz: np.ndarray, v_xy: float = 4.0, psi: float = 0.0):
        self.x[:] = [pos_xyz[0], pos_xyz[1], pos_xyz[2], v_xy, psi, 0.0, 0.0, 0.0]
        self.P = np.eye(8, dtype=float) * 0.25

    def predict(self, dt: float = None) -> np.ndarray:
        """
        Propagates the 3D curvilinear CTRA state and covariance (Eqs. 4-6 in manuscript).
        """
        if dt is None:
            dt = self.dt

        px, py, pz, v, psi, w, a, vz = self.x
        F = np.eye(8, dtype=float)

        if abs(w) > 1e-4:
            psi_new = psi + w * dt
            sin_p, cos_p = math.sin(psi), math.cos(psi)
            sin_pn, cos_pn = math.sin(psi_new), math.cos(psi_new)
            v_new = v + a * dt

            dx = (v_new / w) * sin_pn - (v / w) * sin_p + (a / (w * w)) * (cos_pn - cos_p)
            dy = -(v_new / w) * cos_pn + (v / w) * cos_p + (a / (w * w)) * (sin_pn - sin_p)

            F[0, 3] = (sin_pn - sin_p) / w
            F[0, 4] = (v_new / w) * cos_pn - (v / w) * cos_p - (a / (w * w)) * (sin_pn - sin_p)
            F[1, 3] = (-cos_pn + cos_p) / w
            F[1, 4] = (v_new / w) * sin_pn - (v / w) * sin_p + (a / (w * w)) * (cos_pn - cos_p)
        else:
            sin_p, cos_p = math.sin(psi), math.cos(psi)
            dist = v * dt + 0.5 * a * dt * dt
            dx = dist * cos_p
            dy = dist * sin_p
            psi_new = psi
            v_new = v + a * dt

            F[0, 3] = dt * cos_p
            F[0, 4] = -dist * sin_p
            F[1, 3] = dt * sin_p
            F[1, 4] = dist * cos_p

        F[0, 6] = 0.5 * dt * dt * math.cos(psi)
        F[1, 6] = 0.5 * dt * dt * math.sin(psi)
        F[2, 7] = dt
        F[3, 6] = dt
        F[4, 5] = dt

        self.x[0] = px + dx
        self.x[1] = py + dy
        self.x[2] = pz + vz * dt
        self.x[3] = v_new
        self.x[4] = psi_new

        self.P = F @ self.P @ F.T + self.Q
        return self.x[:3].copy()

    def update(self, z_xyz: np.ndarray) -> np.ndarray:
        """
        Executes measurement update when target bounding box is visible.
        During total occlusion (z_xyz is None), measurement update is gated off.
        """
        if z_xyz is None:
            return self.x[:3].copy()

        y = z_xyz - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(8) - K @ self.H
        self.P = I_KH @ self.P
        return self.x[:3].copy()
