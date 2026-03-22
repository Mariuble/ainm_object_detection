# NM i AI 2026 — Object Detection Summary

## Competition Overview
- **Task**: Detect and classify grocery products on store shelves (NorgesGruppen data)
- **Duration**: March 19 18:00 – March 22 15:00 CET (69 hours)
- **Scoring**: `Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5`
- **Final Score**: **0.8938** (ensemble submission)

## Dataset
- **248 shelf images** (2000×1500 px), ~22,731 COCO bbox annotations
- **356 categories** (IDs 0–355), category 355 = `unknown_product`
- **4 store sections**: Egg, Frokost, Knekkebrod, Varmedrikker
- **Class imbalance**: 74 classes with <5 annotations, top class has 422
- **Category 300**: Empty name `""` in annotations — caused bugs in data.yaml generation
- **327 product reference images** (1,582 total photos): front/back/left/right/top/bottom per product

## Sandbox Constraints
- NVIDIA L4 GPU, 24 GB VRAM, 8 GB RAM, 4 vCPU
- 300 second timeout, no network
- Pre-installed: ultralytics==8.1.0, onnxruntime-gpu 1.20.0, ensemble-boxes 1.0.9
- **YOLO11 NOT supported** — only YOLOv8 via ultralytics 8.1.0
- Blocked imports: os, sys, subprocess, pickle, yaml, etc.
- Max zip: 420 MB uncompressed, max 3 weight files

## Key Technical Decisions

### ONNX Export (Critical)
The sandbox only has ultralytics==8.1.0 which doesn't support YOLO11. All YOLO11 models had to be exported to ONNX format with `opset=17` and inference done via onnxruntime with CUDAExecutionProvider. This was the most important architectural decision — without it, no YOLO11 model would load in the sandbox.

### COCO to YOLO Label Conversion
Labels were pre-converted from COCO format (`[x_min, y_min, w, h]` absolute pixels) to YOLO format (`[class_id, x_center, y_center, w, h]` normalized). Verified correct by comparing first annotations of `img_00001.jpg` — exact match.

### The nc=356 Bug
The `create_data_yaml` function skipped empty lines in `classes.txt`, dropping category 300 (which has an empty name `""`). This caused `nc=355` instead of `nc=356`, making class 355 (`unknown_product`) invalid. Fix: use dict format `{i: name for i, name in enumerate(classes)}` which preserves empty strings.

### Product Reference Images
Created a script (`scripts/add_product_images.py`) to convert 1,577 product reference photos into YOLO training data with full-image bounding boxes (`class_id 0.5 0.5 1.0 1.0`). 328/329 products matched COCO categories by name. However, these images ultimately **hurt more than helped** — clean single-product photos on white backgrounds are very different from dense shelf scenes, potentially confusing the model about what product detection looks like.

### Image Resolution
Original shelf images are 2000×1500 px. Training at imgsz=640 (first Colab run) lost too much detail for small products. Switching to imgsz=1280 was the single biggest improvement (mAP50: 0.494 → 0.715). Testing imgsz=1600 showed marginally better mAP50-95 but didn't improve mAP50.

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

### Submissions

| # | Type | Models | Score |
|---|---|---|---|
| 1 | Single model | Run 3 (YOLO11l) | **0.8845** |
| 2 | Ensemble (2 models) | Run 3 + Run 8 (YOLO11l + YOLO11x) | **0.8938** ⭐ |
| 3 | Ensemble (3 models) | Run 3 + Run 8 + Run 10, low conf threshold | 0.6785 |

## What Worked

1. **imgsz=1280** — massive improvement over 640 for small product detection
2. **YOLO11l** — sweet spot between capacity and training efficiency for 356 classes on 248 images
3. **batch=8** — smaller batch with more gradient noise helped generalization on this tiny dataset
4. **cls=0.5 (default)** — the default classification loss weight was optimal; both higher (1.5) and lower (0.3) hurt
5. **Ensemble of two different architectures** — YOLO11l + YOLO11x with WBF gave +0.93% over single model
6. **ONNX inference** — reliable, GPU-accelerated, works in the sandbox regardless of ultralytics version
7. **A100 GPU on GCP** — fast iteration (~11s/epoch for 248 images at 1280)

## What Didn't Work

1. **Product reference images** — full-image bounding boxes of isolated products confused the detector trained on dense shelf scenes
2. **cls=1.5** — boosting classification loss weight hurt detection accuracy (which is 70% of the score)
3. **batch=16** — less gradient noise led to worse generalization on this small dataset
4. **YOLO26l** — newer architecture didn't outperform YOLO11l on this task
5. **YOLO11x** — extra-large model didn't improve over YOLO11l, likely overfitting the small dataset
6. **imgsz=1600** — diminishing returns over 1280, and slower training
7. **copy_paste augmentation** — didn't improve results
8. **CONF_THRESH=0.01** — too many false positives destroyed mAP in the 3-model ensemble submission
9. **cls=0.3** — lower classification weight didn't help either

## Key Learnings

### Val mAP is Unreliable with Small Val Sets
With only 13 validation images, val mAP50 varied wildly between runs (0.788 to 0.935) even with similar configurations. The random train/val split had more impact on reported val mAP than actual model quality. The real metric was the submission leaderboard score.

### Data Quality > Data Quantity
248 shelf images with 22,731 dense annotations provided more useful training signal than adding 1,577 clean product photos. The domain mismatch between isolated product shots and real shelf scenes was too large.

### Default Hyperparameters Are Often Optimal
For the classification loss weight (cls), the default value of 0.5 outperformed both higher (1.5) and lower (0.3) values. The YOLO defaults are well-tuned for general object detection.

### Smaller Batch = Better Generalization on Small Datasets
batch=8 consistently outperformed batch=16 on this 235-image training set. The additional gradient noise from smaller batches acted as regularization.

### Ensembling Helps, But Carefully
Two-model ensemble with default WBF parameters improved the score (+0.93%). But aggressive parameter tuning (lower confidence threshold, 3 models) backfired catastrophically (0.8938 → 0.6785). Conservative ensembling > aggressive tuning.

## Infrastructure

- **Google Colab** (T4 GPU) — initial training, throttled after ~3 runs
- **GCP Compute Engine** — 2× A100-SXM4-40GB VMs (`yolo-train`, `yolo-train-2`) for parallel training
- **Deep Learning VM image** (`pytorch-2-7-cu128-ubuntu-2204-nvidia-570`) — pre-installed GPU drivers + PyTorch
- Required manual install: `ultralytics`, `libgl1-mesa-glx`, `libglib2.0-0`

## File Structure

```
vibe/
├── train.py        # Training script (split, yaml creation, training, ONNX export)
├── run.py          # Single-model ONNX inference submission
├── best.onnx       # Run 3 YOLO11l model (98 MB)
├── best.pt         # Run 3 YOLO11l weights
└── results.csv     # Run 3 training metrics

ensemble/
├── run.py          # Multi-model ensemble with WBF
├── yolo11l.onnx    # Run 3 model
├── yolo11x.onnx    # Run 8 model
└── yolo11l_1600.onnx # Run 10 model

scripts/
├── add_product_images.py   # Convert product reference images to YOLO format
└── visualize_labels.py     # Draw YOLO bboxes on images for verification
```

## What I Would Do Differently

1. **Fix the random seed for train/val split** — would have made all runs comparable
2. **Submit earlier and more often** — wasted submissions on the aggressive 3-model ensemble
3. **Skip product reference images** — should have focused on shelf-only training from the start
4. **Try test-time augmentation (TTA)** in the single-model run.py — free accuracy boost we never tested in submission
5. **Tune NMS/confidence thresholds** carefully on a proper validation set before submitting
6. **Use all 248 images for training** (no val split) for the final submission, since val metrics were unreliable anyway
