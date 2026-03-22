import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Patch torch.load for torch 2.6+ / ultralytics 8.1.0 compatibility
# torch 2.6 defaults weights_only=True, which breaks ultralytics model loading
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO


def get_detections(model, img_input, device, use_cuda, augment=True):
    """Run model on image (path or array), return (boxes_xyxy, scores, classes)."""
    results = model(
        img_input if isinstance(img_input, str) else img_input,
        device=device, imgsz=1280, conf=0.01, half=use_cuda,
        augment=augment, verbose=False,
    )
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)
    return (
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.conf.cpu().numpy(),
        r.boxes.cls.cpu().numpy(),
    )


def normalize_boxes(boxes, w, h):
    """Normalize xyxy boxes to [0,1] range."""
    norm = boxes.copy()
    norm[:, [0, 2]] /= w
    norm[:, [1, 3]] /= h
    return np.clip(norm, 0, 1)


def slice_image(img_array, slice_wh=(640, 640), overlap=0.2):
    """Generate (crop, offset_x, offset_y) for overlapping tiles."""
    h, w = img_array.shape[:2]
    sw, sh = slice_wh
    step_x = int(sw * (1 - overlap))
    step_y = int(sh * (1 - overlap))
    tiles = []
    for y in range(0, max(h - sh + 1, 1), step_y):
        for x in range(0, max(w - sw + 1, 1), step_x):
            tiles.append((img_array[y:y + sh, x:x + sw], x, y))
    return tiles


def run_sahi_single(model, img_array, device, use_cuda):
    """Sliced inference + full image for one model. Returns normalized boxes."""
    h, w = img_array.shape[:2]
    all_boxes, all_scores, all_labels = [], [], []

    # Full image pass with TTA
    boxes, scores, labels = get_detections(model, img_array, device, use_cuda, augment=True)
    if len(scores) > 0:
        all_boxes.append(normalize_boxes(boxes, w, h))
        all_scores.append(scores)
        all_labels.append(labels.astype(int))

    # Sliced passes (no TTA on tiles — too slow)
    for crop, ox, oy in slice_image(img_array):
        boxes, scores, labels = get_detections(model, crop, device, use_cuda, augment=False)
        if len(scores) == 0:
            continue
        offset_boxes = boxes.copy()
        offset_boxes[:, [0, 2]] = (offset_boxes[:, [0, 2]] + ox) / w
        offset_boxes[:, [1, 3]] = (offset_boxes[:, [1, 3]] + oy) / h
        all_boxes.append(np.clip(offset_boxes, 0, 1))
        all_scores.append(scores)
        all_labels.append(labels.astype(int))

    return all_boxes, all_scores, all_labels


def run_ensemble(models, img_array, device, use_cuda, use_sahi=True):
    """Run all models on image, merge with WBF."""
    from ensemble_boxes import weighted_boxes_fusion

    h, w = img_array.shape[:2]
    all_boxes, all_scores, all_labels = [], [], []

    for model in models:
        if use_sahi and max(w, h) > 1280:
            m_boxes, m_scores, m_labels = run_sahi_single(model, img_array, device, use_cuda)
            all_boxes.extend(m_boxes)
            all_scores.extend(m_scores)
            all_labels.extend(m_labels)
        else:
            boxes, scores, labels = get_detections(model, img_array, device, use_cuda, augment=True)
            if len(scores) > 0:
                all_boxes.append(normalize_boxes(boxes, w, h))
                all_scores.append(scores)
                all_labels.append(labels.astype(int))

    if not all_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        iou_thr=0.5, skip_box_thr=0.01,
    )
    # Denormalize back to pixels
    merged_boxes[:, [0, 2]] *= w
    merged_boxes[:, [1, 3]] *= h
    return merged_boxes, merged_scores, merged_labels


def load_models():
    """Load all available model weight files."""
    weight_files = sorted(Path(".").glob("*.pt"))
    models = []
    for wf in weight_files:
        models.append(YOLO(str(wf)))
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    models = load_models()
    predictions = []

    image_paths = sorted([
        p for p in Path(args.input).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    total = len(image_paths)
    start_time = time.monotonic()
    budget = 280  # 20s safety margin from 300s limit

    for idx, img_path in enumerate(image_paths):
        image_id = int(img_path.stem.split("_")[-1])
        elapsed = time.monotonic() - start_time

        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            h, w = img_array.shape[:2]

            # Decide: SAHI or full-image only based on time budget
            remaining_budget = budget - elapsed
            images_left = total - idx
            per_image_budget = remaining_budget / max(images_left, 1)
            use_sahi = per_image_budget > 3 * len(models)

            boxes, scores, labels = run_ensemble(
                models, img_array, device, use_cuda, use_sahi=use_sahi
            )

            for i in range(len(scores)):
                x1, y1, x2, y2 = boxes[i]
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "bbox": [
                        round(float(x1), 1),
                        round(float(y1), 1),
                        round(float(x2 - x1), 1),
                        round(float(y2 - y1), 1),
                    ],
                    "score": round(float(scores[i]), 4),
                })
        except Exception:
            continue

    # Cap at 50,000 predictions (competition limit), keep highest confidence
    if len(predictions) > 50000:
        predictions.sort(key=lambda p: p["score"], reverse=True)
        predictions = predictions[:50000]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()
