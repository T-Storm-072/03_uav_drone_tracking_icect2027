"""
Ego-Motion Compensated Two-Stage ByteTrack Multi-Object Associator.
Authors: Javeria Dawood, Syed Ali Turab Kazmi, Amber Saeed
Department of Robotics & Intelligent Systems, Bahria University, Islamabad
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Tracklet:
    track_id: int
    bbox_xywh: np.ndarray  # [u, v, w, h]
    depth_est: float
    confidence: float
    age: int = 0
    hits: int = 1
    occluded_frames: int = 0


def warp_bbox_center_with_vio(
    u_prev: float,
    v_prev: float,
    depth_prev: float,
    K: np.ndarray,
    R_k_km1: np.ndarray,
    t_k_km1: np.ndarray,
) -> Tuple[float, float]:
    """
    Projects tracklet image-plane center (u_prev, v_prev) from frame k-1 into frame k
    using the inter-frame camera rigid transform (R_k_km1, t_k_km1) from VIO.
    Implements Equation (2) of the manuscript.
    """
    K_inv = np.linalg.inv(K)
    uv1 = np.array([u_prev, v_prev, 1.0], dtype=float)
    pt_cam_prev = depth_prev * (K_inv @ uv1)
    pt_cam_curr = R_k_km1 @ pt_cam_prev + t_k_km1
    proj = K @ pt_cam_curr
    z_curr = max(1e-4, proj[2])
    return float(proj[0] / z_curr), float(proj[1] / z_curr)


def compute_giou_cost_matrix(
    tracks: List[Tracklet], detections: List[np.ndarray]
) -> np.ndarray:
    """
    Computes hybrid Generalized IoU + normalized center distance cost matrix
    between ego-compensated active tracklets and candidate bounding boxes.
    """
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)), dtype=float)

    cost = np.zeros((len(tracks), len(detections)), dtype=float)
    for i, trk in enumerate(tracks):
        tx, ty, tw, th = trk.bbox_xywh
        tx1, ty1, tx2, ty2 = tx - 0.5 * tw, ty - 0.5 * th, tx + 0.5 * tw, ty + 0.5 * th
        for j, det in enumerate(detections):
            dx, dy, dw, dh = det
            dx1, dy1, dx2, dy2 = dx - 0.5 * dw, dy - 0.5 * dh, dx + 0.5 * dw, dy + 0.5 * dh

            ix1, iy1 = max(tx1, dx1), max(ty1, dy1)
            ix2, iy2 = min(tx2, dx2), min(ty2, dy2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            union = tw * th + dw * dh - inter + 1e-6
            iou = inter / union

            cx1, cy1 = min(tx1, dx1), min(ty1, dy1)
            cx2, cy2 = max(tx2, dx2), max(ty2, dy2)
            c_area = max(1e-6, (cx2 - cx1) * (cy2 - cy1))
            giou = iou - (c_area - union) / c_area

            cost[i, j] = 1.0 - giou
    return cost
