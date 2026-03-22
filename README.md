# NM i AI 2026 — Object Detection Summary

## Team UltraThinkers | March 19–22, 2026

### Final Score: **0.8969** | My best: 0.8938 (ensemble)

### Model Weights: [huggingface.co/norway1994/ainm-object-detection](https://huggingface.co/norway1994/ainm-object-detection)

---

## The Task

Detect and classify grocery products on store shelf images. 248 training images with ~22,731 COCO-format bounding box annotations across 356 product categories. Images range from 481×399 to 5712×4624 pixels, with an average of 92 products per image (very dense shelves). 69-hour competition (March 19–22, 2026).

**Scoring:** `0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5`

## Dataset
- **248 shelf images** (481×399 to 5712×4624 px, most ~2000×1500), ~22,731 COCO bbox annotations
- **356 categories** (IDs 0–355), category 355 = `unknown_product`
- **4 store sections**: Egg, Frokost, Knekkebrod, Varmedrikker
- **~92 products per image** on average (very dense shelves)
- **Class imbalance**: 74 classes with <5 annotations, top class has 422
- **Category 300**: Empty name `""` in annotations — caused bugs in data.yaml generation
- **327 product reference images** (1,582 total photos): front/back/left/right/top/bottom per product

## Sandbox Constraints
- NVIDIA L4 GPU, 24 GB VRAM, 8 GB RAM, 4 vCPU
- 300 second timeout, no network
- **50,000 prediction limit** — submissions with more predictions are truncated
- Pre-installed: ultralytics==8.1.0, onnxruntime-gpu 1.20.0, ensemble-boxes 1.0.9
- **YOLO11 NOT supported** — only YOLOv8 via ultralytics 8.1.0
- Blocked imports: os, sys, subprocess, pickle, yaml, etc.
- Max zip: 420 MB uncompressed, max 3 weight files

---

## Score Progression

| # | Type | Models | Config | Score | Delta |
|---|---|---|---|---|---|
| 1 | Single model (ONNX) | YOLO11l | conf=0.05 | **0.8845** | baseline |
| 2 | Ensemble (2 ONNX) | YOLO11l + YOLO11x | conf=0.05, WBF | **0.8938** | +0.0093 |
| 3 | Ensemble (3 ONNX) | YOLO11l + YOLO11x + YOLO11l-1600 | conf=0.01, WBF | 0.6785 | -0.2153 |

**Submission 3 failure analysis**: Lowering conf to 0.01 with 3 models likely exceeded the **50k prediction limit**, causing truncation and catastrophic mAP loss.

---

## Key Technical Decisions

### ONNX Export (Critical)
The sandbox only has ultralytics==8.1.0 which doesn't support YOLO11. All YOLO11 models had to be exported to ONNX format with `opset=17` and inference done via onnxruntime with CUDAExecutionProvider. This was the most important architectural decision — without it, no YOLO11 model would load in the sandbox.

**Alternative approach (teammate)**: Patch `torch.load` to work with ultralytics 8.1.0 directly:
```python
import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO  # now works with .pt files
```
This allowed teammates to submit YOLOv8 `.pt` files directly without ONNX conversion.

### COCO to YOLO Label Conversion
Labels were pre-converted from COCO format (`[x_min, y_min, w, h]` absolute pixels) to YOLO format (`[class_id, x_center, y_center, w, h]` normalized). Verified correct by comparing first annotations of `img_00001.jpg` — exact match.

### The nc=356 Bug
The `create_data_yaml` function skipped empty lines in `classes.txt`, dropping category 300 (which has an empty name `""`). This caused `nc=355` instead of `nc=356`, making class 355 (`unknown_product`) invalid. Fix: use dict format `{i: name for i, name in enumerate(classes)}` which preserves empty strings.

### Product Reference Images
Created a script (`scripts/add_product_images.py`) to convert 1,577 product reference photos into YOLO training data with full-image bounding boxes (`class_id 0.5 0.5 1.0 1.0`). 328/329 products matched COCO categories by name. However, these images ultimately **hurt more than helped** — clean single-product photos on white backgrounds are very different from dense shelf scenes, potentially confusing the model about what product detection looks like.

### Image Resolution
Original shelf images are up to 5712×4624 px. Training at imgsz=640 (first Colab run) lost too much detail for small products. Switching to imgsz=1280 was the single biggest improvement (mAP50: 0.494 → 0.715). Testing imgsz=1600 showed marginally better mAP50-95 but didn't improve mAP50. Inference at higher resolution than training (e.g. 2048, 3200) was **much worse** (teammate finding).

---

## Training Runs

| # | Model | Data | GPU | Batch | cls | imgsz | Epochs | Best mAP50 | Submission Score | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | YOLO11s | shelf (248) | Colab T4 | 16 | 0.5 | 640 | 200 | 0.494 | — | Baseline, too low resolution |
| 2 | YOLO11m | shelf (248) | Colab T4 | 8 | 0.5 | 1280 | ~150 | 0.715 | — | Resolution boost helped hugely |
| 3 | YOLO11l | shelf (235 train) | A100 | 8 | 0.5 | 1280 | 180 | **0.935** | **0.8845** | Best single model ⭐ |
| 4 | YOLO26l | shelf (235 train) | A100 | 16 | 1.5 | 1280 | 180 | 0.798 | — | Different architecture, underperformed |
| 5 | YOLO11l | combined (1806 train) | A100 | 8 | 1.5 | 1280 | 81/180 | 0.860 | — | Product images didn't help enough |
| 6 | YOLO11l | shelf (235 train) | A100 | 16 | 1.5 | 1280 | ~140/180 | 0.820 | — | cls=1.5 hurt detection |
| 7 | YOLO11l | shelf (235 train) | A100 | 16 | 0.5 | 1280 | ~86/180 | 0.788 | — | batch=16 worse than batch=8 |
| 8 | YOLO11x | shelf (235 train) | A100 | 8 | 0.5 | 1280 | 132/300 | 0.859 | — | Bigger model, didn't beat run 3 |
| 9 | YOLO11l | shelf (235 train) | A100 | 8 | 0.3 | 1280 | 154/300 | 0.828 | — | Lower cls didn't help |
| 10 | YOLO11l | shelf (235 train) | A100 | 4 | 0.5 | 1600 | 138/300 | 0.863 | — | Higher res, marginal improvement |
| 11 | YOLO11l | shelf (235 train) | A100 | 8 | 0.5 | 1280 | 199/300 | 0.863 | — | copy_paste=0.3, no improvement |

---

## What Worked

1. **imgsz=1280** — massive improvement over 640 for small product detection
2. **YOLO11l** — sweet spot between capacity and training efficiency for 356 classes on 248 images
3. **batch=8** — smaller batch with more gradient noise helped generalization on this tiny dataset
4. **cls=0.5 (default)** — the default classification loss weight was optimal; both higher (1.5) and lower (0.3) hurt
5. **Ensemble of two different architectures** — YOLO11l + YOLO11x with WBF gave +0.93% over single model
6. **ONNX inference** — reliable, GPU-accelerated, works in the sandbox regardless of ultralytics version
7. **A100 GPU on GCP** — fast iteration (~11s/epoch for 248 images at 1280)
8. **Simple single-pass inference** — teammate confirmed: complex pipelines (SAHI+TTA+WBF) scored 0.45, same model with simple inference scored 0.85

## What Didn't Work

1. **Product reference images** — full-image bounding boxes of isolated products confused the detector trained on dense shelf scenes
2. **cls=1.5** — boosting classification loss weight hurt detection accuracy (which is 70% of the score)
3. **batch=16** — less gradient noise led to worse generalization on this small dataset
4. **YOLO26l** — newer architecture didn't outperform YOLO11l on this task
5. **YOLO11x** — extra-large model didn't improve over YOLO11l, likely overfitting the small dataset
6. **imgsz=1600** — diminishing returns over 1280, and slower training
7. **copy_paste augmentation** — didn't improve our results (though teammate found copy_paste=0.3 + mixup=0.15 helped with YOLOv8x)
8. **CONF_THRESH=0.01 with ensemble** — too many predictions exceeded 50k limit, destroying mAP
9. **cls=0.3** — lower classification weight didn't help either
10. **SAHI (Slicing Aided Hyper Inference)** — teammate found overlapping tiles + WBF created imprecise averaged boxes
11. **Test-Time Augmentation (TTA)** — teammate found it added noise, inconsistent detections across augmented views
12. **Higher inference resolution** — 2048/3200 at inference with 1280-trained model was much worse (teammate finding)
13. **Agnostic NMS** — suppressed valid overlapping products of different classes (teammate finding)

---

## Key Learnings

### Simple Inference >>> Complex Pipelines
The team's biggest lesson: SAHI + TTA + WBF scored **0.4518**. The exact same model with plain single-pass inference scored **0.8514**. Almost 2× better. Complex pipelines generate overlapping/averaged bounding boxes that hurt IoU matching.

### Val mAP is Unreliable with Small Val Sets
With only 13 validation images, val mAP50 varied wildly between runs (0.788 to 0.935) even with similar configurations. The random train/val split had more impact on reported val mAP than actual model quality. The real metric was the submission leaderboard score.

### 50,000 Prediction Limit
The competition caps predictions at 50k. Low confidence thresholds (conf=0.01) with ensemble pipelines easily exceed this. Must either raise conf or cap predictions by keeping top-k by confidence. This is likely what killed our 3-model ensemble (0.6785).

### Confidence Threshold Matters (With Caveats)
Teammate found lower conf = more predictions = better recall = higher mAP (conf=0.01 scored 0.8969 vs conf=0.25 at 0.8911). But only works if you stay under the 50k prediction limit. With ensemble + low conf, we exceeded it.

### Data Quality > Data Quantity
248 shelf images with 22,731 dense annotations provided more useful training signal than adding 1,577 clean product photos. The domain mismatch between isolated product shots and real shelf scenes was too large.

### Default Hyperparameters Are Often Optimal
For the classification loss weight (cls), the default value of 0.5 outperformed both higher (1.5) and lower (0.3) values. The YOLO defaults are well-tuned for general object detection.

### Smaller Batch = Better Generalization on Small Datasets
batch=8 consistently outperformed batch=16 on this 235-image training set. The additional gradient noise from smaller batches acted as regularization.

### Ensembling Helps, But Carefully
Two-model ensemble with default WBF parameters improved the score (+0.93%). But aggressive parameter tuning (lower confidence threshold, 3 models) backfired catastrophically (0.8938 → 0.6785). Conservative ensembling > aggressive tuning.

---

## Infrastructure

### Training Speed Comparison

| GPU | Batch size @ 1280 | Time per epoch (248 imgs) | 200 epochs |
|-----|-------------------|--------------------------|------------|
| Colab T4 | 8 | ~55s | ~3 hrs |
| GCP A100 | 8 | ~11s | ~37 min |

### VMs Used

- **Google Colab** (T4 GPU) — initial training, throttled after ~3 runs
- **GCP `yolo-train`** (A100-SXM4-40GB, us-central1-f) — primary training
- **GCP `yolo-train-2`** (A100-SXM4-40GB, us-central1-f) — parallel experiments
- **Deep Learning VM image** (`pytorch-2-7-cu128-ubuntu-2204-nvidia-570`) — pre-installed GPU drivers + PyTorch
- Required manual install: `ultralytics`, `libgl1-mesa-glx`, `libglib2.0-0`

---

## Timeline

| When | What |
|------|------|
| Mar 19, evening | Downloaded data, set up repo, initial exploration |
| Mar 20 | YOLO11s on Colab (0.494 mAP50), identified imgsz=640 as bottleneck |
| Mar 21, morning | YOLO11m at imgsz=1280 on Colab (0.715), Colab throttled |
| Mar 21, afternoon | Created GCP A100 VM, YOLO11l training (0.935 val mAP50) |
| Mar 21, evening | First submission: 0.8845, started parallel experiments |
| Mar 21, night | Overnight runs: YOLO11x, cls=0.3, combined data |
| Mar 22, morning | Ensemble submission: 0.8938, failed 3-model ensemble: 0.6785 |
| Mar 22, afternoon | Final experiments, cleanup, documentation |

---

## Tools & Stack

- **Training:** ultralytics (latest for YOLO11), PyTorch 2.7.1, Python 3.10
- **Inference:** onnxruntime-gpu 1.20.0 (ONNX export for sandbox compatibility)
- **Infrastructure:** Google Cloud Platform (A100 GPU VMs), Google Colab (T4)
- **Development:** Claude Code (AI pair programming)
- **Ensembling:** ensemble-boxes (Weighted Box Fusion)

---

## Project Structure

```
├── README.md                       # This file
├── LICENSE                         # MIT
├── train.py                        # GCP training script
├── task/                           # Competition documentation
├── scripts/
│   ├── add_product_images.py       # Convert product reference images to YOLO format
│   └── visualize_labels.py         # Draw YOLO bboxes on images for verification
├── submissions/
│   ├── best-ensemble/run.py        # 0.8938 — 2-model ONNX ensemble with WBF
│   ├── best-single/run.py          # 0.8845 — single ONNX model inference
│   └── teammate-variants/          # Teammate's run.py variants (SAHI, tuned, etc.)
└── experiments/
    ├── 01-yolov8n-baseline/        # Early YOLOv8n runs (8 experiments)
    ├── 02-yolo11s-colab/           # First YOLO11s Colab run (0.494)
    ├── 03-yolo11l-colab/           # Colab notebook export
    └── 04-gcp-runs/                # GCP A100 training results
```

---

## What I Would Do Differently

1. **Fix the random seed for train/val split** — would have made all runs comparable
2. **Submit earlier and more often** — wasted a submission on the aggressive 3-model ensemble
3. **Skip product reference images** — should have focused on shelf-only training from the start
4. **Use simple inference from the start** — team proved complex pipelines hurt badly
5. **Cap predictions at 50k** — would have saved the 3-model ensemble submission
6. **Try conf=0.01 with single model** — teammate's best score used this
7. **Use all 248 images for training** (no val split) for the final submission, since val metrics were unreliable anyway
8. **Use the torch.load patch** instead of ONNX — simpler pipeline, and could have submitted YOLOv8x `.pt` directly
9. **Try enhanced augmentation** (copy_paste=0.3 + mixup=0.15) — worked for teammates with YOLOv8x

---

*Built with Claude Code during NM i AI 2026. From 0.494 to 0.8938 in 48 hours.*
