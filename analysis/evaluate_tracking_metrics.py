"""
Evaluation Script for Gazebo UAV Tracking Telemetry and Ablation Benchmarks.
Computes 3D Trajectory RMSE (overall and during occlusion windows), VIO ATE_RMSE,
and prints summary tables matching Tables I & II of the ICECT 2027 paper.
"""

import csv
import math
import os


def evaluate_telemetry(csv_path: str):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    sq_err_ekf_all = []
    sq_err_lin_all = []
    sq_err_ekf_occ = []
    sq_err_lin_occ = []
    sq_err_vio = []

    for r in rows:
        gx, gy, gz = float(r["uav_gt_x"]), float(r["uav_gt_y"]), float(r["uav_gt_z"])
        ex, ey, ez = float(r["ekf_vio_x"]), float(r["ekf_vio_y"]), float(r["ekf_vio_z"])
        lx, ly, lz = float(r["linear_kf_x"]), float(r["linear_kf_y"]), float(r["linear_kf_z"])
        cx, cy, cz = float(r["cam_gt_x"]), float(r["cam_gt_y"]), float(r["cam_gt_z"])
        vx, vy, vz = float(r["cam_vio_x"]), float(r["cam_vio_y"]), float(r["cam_vio_z"])
        occ = int(r["is_occluded"])

        e_ekf = (gx - ex) ** 2 + (gy - ey) ** 2 + (gz - ez) ** 2
        e_lin = (gx - lx) ** 2 + (gy - ly) ** 2 + (gz - lz) ** 2
        e_vio = (cx - vx) ** 2 + (cy - vy) ** 2 + (cz - vz) ** 2

        sq_err_ekf_all.append(e_ekf)
        sq_err_lin_all.append(e_lin)
        sq_err_vio.append(e_vio)
        if occ == 1:
            sq_err_ekf_occ.append(e_ekf)
            sq_err_lin_occ.append(e_lin)

    rmse_ekf_all = math.sqrt(sum(sq_err_ekf_all) / len(sq_err_ekf_all))
    rmse_lin_all = math.sqrt(sum(sq_err_lin_all) / len(sq_err_lin_all))
    rmse_ekf_occ = math.sqrt(sum(sq_err_ekf_occ) / len(sq_err_ekf_occ))
    rmse_lin_occ = math.sqrt(sum(sq_err_lin_occ) / len(sq_err_lin_occ))
    ate_vio = math.sqrt(sum(sq_err_vio) / len(sq_err_vio))

    print(f"Evaluated {len(rows)} frames ({len(sq_err_ekf_occ)} occluded frames):")
    print(f"  Proposed 3D CTRA-EKF + VIO Overall RMSE : {rmse_ekf_all:.3f} m")
    print(f"  Proposed 3D CTRA-EKF + VIO Occlusion RMSE: {rmse_ekf_occ:.3f} m")
    print(f"  Baseline Linear KF Occlusion RMSE        : {rmse_lin_occ:.3f} m")
    print(f"  Monocular VIO Camera ATE_RMSE            : {ate_vio:.4f} m")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(base_dir, "data", "gazebo_uav_tracking_telemetry.csv")
    evaluate_telemetry(csv_file)
