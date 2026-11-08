"""
YOLOv8n-P2 Small-UAV Detection Module with Wise-IoU v3 Loss and C2f-Ghost Blocks.
Authors: Javeria Dawood, Syed Ali Turab Kazmi, Amber Saeed
Department of Electrical Engineering, Bahria University, Islamabad
"""

import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class BoundingBox:
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float
    class_id: int = 0

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        x1 = self.x_center - 0.5 * self.width
        y1 = self.y_center - 0.5 * self.height
        x2 = self.x_center + 0.5 * self.width
        y2 = self.y_center + 0.5 * self.height
        return (x1, y1, x2, y2)


def compute_wise_iou_loss(
    pred_box: BoundingBox,
    gt_box: BoundingBox,
    alpha: float = 1.9,
    delta: float = 3.0,
    eps: float = 1e-7,
) -> float:
    """
    Computes Wise-IoU (WIoU) bounding-box regression loss with dynamic non-monotonic
    focusing coefficient for small aerial targets (8x8 to 24x24 px).
    """
    px1, py1, px2, py2 = pred_box.to_xyxy()
    gx1, gy1, gx2, gy2 = gt_box.to_xyxy()

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    pred_area = max(eps, pred_box.width * pred_box.height)
    gt_area = max(eps, gt_box.width * gt_box.height)
    union_area = pred_area + gt_area - inter_area + eps

    iou = inter_area / union_area
    l_iou = 1.0 - iou

    # Smallest enclosing box (detached in backprop)
    wg = max(px2, gx2) - min(px1, gx1) + eps
    hg = max(py2, gy2) - min(py1, gy1) + eps

    center_dist_sq = (pred_box.x_center - gt_box.x_center) ** 2 + (
        pred_box.y_center - gt_box.y_center
    ) ** 2
    r_wiou = math.exp(center_dist_sq / (wg ** 2 + hg ** 2))

    # Dynamic non-monotonic focusing factor beta
    beta = l_iou / max(eps, 0.15)
    focusing_coeff = beta / (delta * (alpha ** (beta - delta)))

    return float(focusing_coeff * r_wiou * l_iou)


class YOLOv8nP2Detector:
    """
    Wrapper for the 4-scale YOLOv8n-P2 detector (strides 4, 8, 16, 32 -> 160x160 P2 head).
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (640, 640),
        tau_high: float = 0.50,
        tau_low: float = 0.15,
    ):
        self.input_size = input_size
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.strides = (4, 8, 16, 32)

    def partition_detections(
        self, detections: List[BoundingBox]
    ) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        """
        Splits raw P2-P5 detections into high-confidence and low-confidence pools
        for two-stage ByteTrack association.
        """
        d_high = [d for d in detections if d.confidence >= self.tau_high]
        d_low = [
            d for d in detections if self.tau_low <= d.confidence < self.tau_high
        ]
        return d_high, d_low
