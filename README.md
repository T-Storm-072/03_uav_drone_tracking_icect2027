# Lightweight AI-Based Small Drone Detection, Visual Tracking, and Short-Term Trajectory Prediction Under Occlusion and Camera Motion

Official repository for our paper prepared for the **5th International Conference on Engineering & Computing Technologies (ICECT 2027)** at NUML, Islamabad, Pakistan.

## Authors
* **Javeria Dawood** — Department of Electrical Engineering, Bahria School of Engineering and Applied Sciences (BSEAS), Bahria University, Islamabad (`javeriadawood433@gmail.com`)
* **Syed Ali Turab Kazmi** — Department of Electrical Engineering, Bahria School of Engineering and Applied Sciences (BSEAS), Bahria University, Islamabad (`syedaliturab072@gmail.com`)
* **Amber Saeed** — Department of Electrical Engineering, Bahria School of Engineering and Applied Sciences (BSEAS), Bahria University, Islamabad (`ambersaeed2005@gmail.com`)

---

## Overview
Tracking agile micro-UAVs ($8 	imes 8$ to $24 	imes 24$ pixels) from a moving camera platform introduces four coupled failure modes: spatial feature loss during strided downsampling, confidence drops in cluttered urban/foliage backgrounds, multi-second full visual occlusions behind structures, and apparent bounding-box jumps caused by camera pan/tilt ego-motion.

This repository implements our four-stage aerial monitoring pipeline evaluated in **Gazebo Harmonic 8.15** and **ROS 2 Humble**:
1. **YOLOv8n-P2 Small-Drone Detector (`detector/yolov8n_p2_detector.py`)**: Incorporates a high-resolution $160 	imes 160$ stride-4 $P_2$ detection head, C2f-Ghost blocks (`2.84 M` parameters), and Wise-IoU (WIoU) bounding-box regression loss.
2. **Tightly Coupled VIO Ego-Motion Compensation (`tracker/bytetrack_ego_associator.py`)**: Fuses 200 Hz IMU pre-integration with 30 Hz monocular visual features ($\mathbf{T}_w^{c_t} \in SE(3)$) to project tracklet priors into the current camera frame before data association.
3. **Confidence-Stratified ByteTrack Association (`tracker/bytetrack_ego_associator.py`)**: Matches high-confidence detections ($	au_{	ext{high}} = 0.50$) in Stage 1 and rescues partially occluded/motion-blurred drones ($	au_{	ext{low}} = 0.15$) in Stage 2.
4. **3D CTRA Extended Kalman Filter (`estimator/ctra_ekf_vio.py`)**: Propagates an 8-state 3D Constant-Turn-Rate and Acceleration (CTRA) state vector across $1.0	ext{--}3.0	ext{ s}$ blind occlusion intervals.

---

## Repository Structure
```text
03_uav_drone_tracking_icect2027/
├── detector/
│   └── yolov8n_p2_detector.py        # YOLOv8n-P2 head & Wise-IoU loss implementation
├── tracker/
│   └── bytetrack_ego_associator.py   # VIO homography box warping & 2-stage ByteTrack association
├── estimator/
│   └── ctra_ekf_vio.py               # 8-state 3D CTRA Extended Kalman Filter for occlusion bridging
├── worlds/
│   └── uav_aerial_surveillance.sdf   # Gazebo Harmonic 8.15 aerial testbed world (30 Hz RGB + 200 Hz IMU)
├── data/
│   ├── gazebo_uav_tracking_telemetry.csv # 30 Hz recorded 3D trajectory, VIO pose & occlusion logs
│   └── ablation_summary.csv          # Summary metrics across configurations M0-M3
├── analysis/
│   └── evaluate_tracking_metrics.py  # Script to evaluate 3D RMSE and VIO ATE_RMSE from telemetry
├── figures/                          # High-resolution architectural and experimental plots
└── paper/
    ├── ICECT2027_Drone_Tracking_Paper.pdf
    └── ICECT2027_Drone_Tracking_Paper.tex
```

---

## Key Experimental Results

### 1. Small-Drone Detection Performance ($640 	imes 640$, 14,400 Frames)
| Model | Backbone / Head | Params (M) | GFLOPs | Precision (%) | Recall (%) | F1 (%) | $	ext{AP}_{50}$ (%) | $	ext{mAP}_{50:95}$ (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline YOLOv8n | CSPDarknet (P3–P5) | 3.01 | 8.1 | 81.4 | 73.2 | 77.1 | 78.6 | 46.2 |
| SOD-YOLO | RFCBAM + BSSI-FPN | 2.65 | 9.8 | 88.2 | 83.6 | 85.8 | 87.8 | 54.1 |
| Lightweight YOLO-SR | C2f-SA + DDHead | 4.02 | 12.1 | 88.7 | 84.1 | 86.3 | 88.2 | 55.0 |
| **Proposed YOLOv8n-P2** | **C2f-Ghost + P2 Head + WIoU** | **2.84** | **9.2** | **92.3** | **88.7** | **90.5** | **91.4** | **58.9** |

### 2. End-to-End Tracking, VIO & CTRA-EKF Ablation ($R_1	ext{--}R_4$)
| Config | Pipeline Configuration | MOTA (%) | IDF1 (%) | HOTA (%) | IDSW $\downarrow$ | RMSE@1s (m) | RMSE@2s (m) | RMSE@3s (m) | FPS (RTX 3060) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $M_0$ | YOLOv8n + Standard Kalman SORT (No VIO) | 64.2 | 58.5 | 52.1 | 38 | 0.79 | 1.98 | 3.42 | 196.1 |
| $M_1$ | YOLOv8n-P2 + ByteTrack (2D Linear KF, No VIO) | 76.8 | 74.1 | 65.4 | 22 | 0.68 | 1.74 | 3.05 | 178.6 |
| $M_2$ | YOLOv8n-P2 + ByteTrack + VIO Ego-Comp + Lin-KF | 84.5 | 83.2 | 73.8 | 14 | 0.48 | 1.15 | 2.09 | 135.1 |
| **$M_3$** | **Proposed: YOLOv8n-P2 + ByteTrack + VIO + 3D CTRA-EKF** | **89.4** | **91.2** | **81.6** | **8** | **0.19** | **0.36** | **0.61** | **128.2** |

---

## Running the Telemetry Evaluation
To verify the trajectory tracking RMSE and VIO ego-motion metrics from the recorded Gazebo telemetry log:
```bash
python3 analysis/evaluate_tracking_metrics.py
```
