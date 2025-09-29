# MVMono3D 📷📷📷📷 ➜ 🧠 ➜ 🧊 Point Clouds & 3D Skeletons

**Multi-View Monochrome (Global-Shutter, IR, HW-Sync) 3D Reconstruction with AI Geometry & Occlusion Reasoning**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](#-license) [![Status: v0-planning](https://img.shields.io/badge/status-v0--planning-yellow)]() [![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green)]()

> **Goal:** Real-time, **metric 3D** from **4× mono cameras** with **IR** and **hardware trigger**.
> **New twist:** an **AI layer that understands geometry and occlusions**, speeding up triangulation, filling gaps, and stabilizing motion under partial view.

---

## ✨ Why MVMono3D?

* **Metric & consistent**: HW-sync + calibrated mono = reliable **mm-level** 3D (scene & hands).
* **AI for geometry**: priors about **shape, kinematics, and contact** guide reconstruction, cut outliers, and **accelerate convergence**.
* **Occlusion reasoning**: learn typical **self-occlusion** and **object occlusion** patterns; keep poses **stable** and **recover fast**.
* **Efficient**: **ROI point clouds** + **3D keypoints** → less GPU/VRAM, lower latency.
* **Interoperable**: exports **Pose API** compatible with downstream consumers (e.g., MotionCoder, Blender/Unreal).

---

## 🏗️ High-Level Architecture

```
Cameras (4× Mono, GS, IR, HW-Trigger)
      │  RAW8/10/12 + timestamps (PTP)
      ▼
Capture & Sync  ──►  Rectify / Undistort  ──►  2D Pose & ROI (per view)
      │                         │
      └─────► Exposure/IR ctrl  └────► Feature tracks / descriptors
                                      │
                                      ▼
           Multi-View Association & Triangulation (robust, BA)
                         │
                         ├─► 3D Keypoints (hands/body/objects)
                         ├─► ROI Point Cloud (sparse/TSDF/voxel)
                         └─► Confidence, Occlusion masks
                                      │
                         AI Geometry & Occlusion Module
                           (priors, temporal inference, imputation)
                                      │
                                      ▼
                         Outputs: { keypoints3d, roi_pc, poses, metrics }
```

**Core ideas**

* **Geometry priors** (hand kinematics, object primitives, local planarity/curvature).
* **Occlusion model** (view-dependent visibility, temporal persistence).
* **Temporal inference** (filtering, constraints, short-horizon prediction).

---

## 🔍 What’s New vs. Classic Pipelines

| Area               | Classic (e.g., Anipose/MediaPipe + Triangulation) | **MVMono3D** (proposed)                                     |
| ------------------ | ------------------------------------------------- | ----------------------------------------------------------- |
| Occlusion handling | Heuristics, NMS, simple smoothing                 | **Learned occlusion states**, visibility priors, imputation |
| Geometry knowledge | Generic lifting/BA                                | **Shape/kinematic priors**, local **planar/curved** patches |
| Speed vs. accuracy | Trade-off heavy                                   | **ROI-first** + priors → faster **and** more stable         |
| Data products      | 3D keypoints                                      | **3D keypoints + ROI point cloud**, confidence maps         |
| Sync               | Often soft                                        | **Hard HW-sync**, PTP timestamps                            |
| Lighting           | Visible RGB only                                  | **IR-assisted** short exposure (less motion blur)           |

> We *reference* Anipose/MediaPipe as inspiration for components (2D pose, association), but re-architect for **occlusion robustness** and **geometry-aware speed**.

---

## 📦 Repo Layout (proposed)

```
mvmono3d/
├─ apps/
│  ├─ capture/                 # GStreamer/Aravis/Spinnaker; trigger/PTP; IR control
│  ├─ recon_service/           # gRPC/WebSocket server (triangulation+AI inference)
│  └─ viewer/                  # Vulkan/GL viewer (ROI PC & keypoints overlay)
├─ mvmono3d/
│  ├─ calib/                   # ChArUco, intrinsics/extrinsics, distortion models
│  ├─ detect2d/                # 2D pose nets, ROI detector, descriptors
│  ├─ assoc/                   # tracklets, cross-view matching, epipolar checks
│  ├─ triang/                  # linear/robust triang, reproj filters, local BA (Ceres)
│  ├─ ai/                      # geometry priors, occlusion model, temporal inference
│  ├─ pc/                      # ROI voxel/TSDF, downsample/normal/curvature
│  ├─ io/                      # MsgPack/FlatBuffers, DMABUF/zero-copy paths
│  └─ utils/                   # timing, profiles, metrics
├─ configs/                    # camera, calib, runtime, model configs
├─ scripts/                    # record, calibrate, train, export, benchmark
├─ docs/                       # guides, diagrams, model card, privacy
└─ tests/
```

---

## 🚀 Quick Start (Prototype Path)

### 1) Hardware (minimum)

* **4× mono cameras** (global shutter), 2–5 MP, ≥120 FPS
* **HW trigger** (shared line) + **IR illumination**
* Sufficient **USB3/GigE** bandwidth; separate controllers or PoE NIC
* **Sync**: PTP or trigger fanout; stable mounting & known baseline

### 2) Software

* **Ubuntu 22.04 LTS** (recommended)
* **Python 3.11+**, **C++17** toolchain
* **CUDA 12.x** (NVIDIA) for acceleration
* **OpenCV**, **Ceres Solver**, **PyTorch**, **Open3D**, **GStreamer** (DMA-BUF where available)

```bash
git clone https://github.com/kostukovic/mvmono3d.git
cd mvmono3d
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

### 3) Calibrate (ChArUco)

```bash
# capture synchronized boards
python scripts/record_charuco.py --out data/calib/raw/ --cams 0,1,2,3

# intrinsics + extrinsics
python scripts/calibrate.py \
  --in data/calib/raw \
  --out data/calib/solution \
  --board A4_charuco_5x7.json
```

### 4) Run the Reconstruction Service

```bash
python apps/recon_service/main.py \
  --config configs/runtime/quad_mono_ir.yaml
```

### 5) Viewer (optional)

```bash
python apps/viewer/main.py --endpoint ws://127.0.0.1:8787
```

---

## 🧠 AI Modules (Geometry & Occlusion)

### Geometry Priors

* **Hands/body:** bone length constraints, joint limits, smoothness.
* **Object patches:** **local planarity/curvature** priors; primitive fits (plane, cylinder, sphere) around ROIs.
* **Contact hints:** if fingertips near a plane, prefer **contact-consistent** hypotheses.

### Occlusion Model

* **View-dependent visibility** via predicted **occlusion masks** (per joint/patch, per camera).
* **Temporal persistence**: missing joints persist briefly with **uncertainty decay**.
* **Imputation**: short-horizon prediction guided by kinematics & velocity.
* **Confidence fusion**: re-weigh views using predicted visibility & reprojection error.

### Learning Modes

* **Supervised:** labeled 2D/3D keypoints & masks (lab scenes).
* **Self-Supervised pretraining:**

  * **Multi-view consistency** (epipolar/triangulation loss)
  * **Masked-frame/joint modeling** (recover hidden joints)
  * **Photometric/feature-cycle** within ROI patches
* **Fine-tune** on task (hands/objects), minimal labels.

---

## 🧰 Data & Formats

* **2D detections:** `T × V × K × 2/3` (time, view, keypoint, xy/(xyc))
* **3D keypoints:** `T × K × 3`, confidences `T × K`
* **ROI point cloud:** sparse voxels or TSDF tiles (local neighborhoods)
* **Occlusion masks:** `T × V × K ∈ {visible, self, object, unknown}`
* **Serialization:** **FlatBuffers/MsgPack** with timestamps; zero-copy shared memory where possible.

---

## ⏱️ Latency Budget (target)

* **Capture → 2D pose**: 4–8 ms (light model, quantized)
* **Assoc + Triang**: 1–3 ms (batch, SIMD)
* **AI geo/occ**: 1–2 ms (tiny encoder)
* **ROI PC update**: 2–5 ms (GPU vox/TSDF, limited volume)
* **Total**: **≤ 16–24 ms** (≈ 40–60 Hz E2E), hardware-dependent

---

## 🔌 APIs

### Pose Stream (server → client)

```json
{
  "t_ns": 1234567890123,
  "keypoints3d": [[x,y,z,conf], ...],
  "roi_pc": "draco://…", 
  "occlusion": ["visible","self","object",...],
  "metrics": {"reproj_rmse": 0.61, "lat_ms": 18.4}
}
```

### Control (client → server)

```json
{ "cmd": "set_roi", "value": {"center":[0,0,0], "size":[0.3,0.3,0.3]} }
```

---

## 🧪 Training Recipes (examples)

**Supervised**

```bash
python scripts/train_supervised.py \
  --cfg configs/train/hands_supervised.yaml \
  DATA.TRAIN data/hands_lab/ DATA.VAL data/hands_val/
```

**Self-Supervised Pretraining**

```bash
python scripts/pretrain_ssl.py \
  --cfg configs/train/ssl_multiview.yaml \
  DATA.UNLABELED data/captures_unlabeled/
```

**Fine-Tune**

```bash
python scripts/finetune.py \
  --cfg configs/train/finetune_hands.yaml \
  DATA.LABELED data/hands_lab_small/
```

---

## 🧭 Implementation Guide (Leitfaden)

1. **Get Sync Right** 🔧
   HW trigger + stable exposure; verify per-frame simultaneity.
2. **Nail Calibration** 🎯
   ChArUco, reproj error logs, temperature drift checks.
3. **Start Thin** 🪶
   Light 2D model, limited joints/ROIs → hit FPS targets first.
4. **Add Priors** 🧩
   Enable kinematic limits, simple plane fits → observe stability gains.
5. **Teach Occlusion** 🫥
   Label short sequences; pretrain occlusion masks; add temporal imputation.
6. **Tight Loop** ♻️
   Profile → prune → quantize; move hot paths to C++/CUDA.
7. **Measure** 📊
   Track p95 latency, reproj RMSE, dropouts, recovery time after occlusion.
8. **Export** 📦
   ONNX/TensorRT for deploy; keep a CPU fallback for dev.

---

## 🔒 Privacy & Ethics (short)

* On-prem by default; no cloud requirement.
* No biometric identity; only **pose/geometry content**.
* Optional recording with **consent**, retention policy, and local encryption.
* See `docs/privacy.md`, `docs/model_card.md`, `docs/toms.md`.

---

## 🗺️ Roadmap

* **v0.1** Capture + calib + basic triangulation + viewer
* **v0.2** AI occlusion masks + temporal imputation (SSL pretrain)
* **v0.3** Geometry priors (kinematics + planar/curved patches)
* **v0.4** ROI TSDF/vox + fast GPU fusion
* **v0.5** Exporters (BVH/FBX/JSON), ONNX/TensorRT
* **v0.6** Integration demos (MotionCoder, Blender/Unreal bridges)

---

## 🧾 License

**Apache-2.0** — permissive for FOSS and commercial use. See `LICENSE`.

---

## 🙌 Acknowledgments

* Prior art & inspiration: **Anipose**, **MediaPipe**, **OpenCV**, **Open3D**, **Ceres**.
* Community researchers in **multi-view geometry**, **pose**, and **occlusion reasoning**.

---

**MVMono3D** aims to make **precise, fast, and occlusion-robust 3D** a practical default for real workflows.
PRs, issues, design notes — all welcome! 🚀
