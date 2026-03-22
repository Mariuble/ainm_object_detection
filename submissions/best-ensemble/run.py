"""
Ensemble ONNX inference for grocery shelf object detection.
Runs multiple YOLO models exported to ONNX, merges predictions with WBF.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_SIZE = 1280
CONF_THRESH = 0.01
WBF_IOU_THRESH = 0.55
WBF_SKIP_THRESH = 0.005
MAX_DET = 600
PAD_VALUE = 114
DETECTION_ONLY = False
TIME_BUDGET = 280  # seconds, 20s safety margin from 300s limit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def letterbox(img_array, new_shape=INPUT_SIZE):
    h, w = img_array.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = np.array(
        Image.fromarray(img_array).resize((new_w, new_h), Image.LANCZOS)
    )

    pad_w = (new_shape - new_w) // 2
    pad_h = (new_shape - new_h) // 2
    canvas = np.full((new_shape, new_shape, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized

    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)
    return blob, scale, pad_w, pad_h


def decode_yolo_output(output, scale, pad_w, pad_h, orig_w, orig_h):
    """Decode raw YOLO ONNX output into boxes, scores, labels in original image coords."""
    pred = output[0]

    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    boxes_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:]

    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]

    mask = confidences >= CONF_THRESH
    boxes_cxcywh = boxes_cxcywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(boxes_cxcywh) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # Convert cx,cy,w,h to x1,y1,x2,y2
    boxes_xyxy = np.empty_like(boxes_cxcywh)
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    # Map back to original image coords
    boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_w) / scale
    boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_h) / scale
    boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_w) / scale
    boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_h) / scale

    # Clip
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    return boxes_xyxy, confidences, class_ids.astype(int)


def normalize_boxes(boxes_xyxy, w, h):
    """Normalize xyxy boxes to [0,1] for WBF."""
    norm = boxes_xyxy.copy()
    norm[:, [0, 2]] /= w
    norm[:, [1, 3]] /= h
    return np.clip(norm, 0, 1)


def image_id_from_path(p):
    return int(p.stem.split("_")[-1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    script_dir = Path(__file__).resolve().parent

    # Load all ONNX models
    onnx_files = sorted(script_dir.glob("*.onnx"))
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sessions = []
    for onnx_path in onnx_files:
        session = ort.InferenceSession(str(onnx_path), sess_options, providers=providers)
        input_name = session.get_inputs()[0].name
        sessions.append((session, input_name))

    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    all_predictions = []
    start_time = time.monotonic()
    total = len(image_paths)

    for idx, img_path in enumerate(image_paths):
        image_id = image_id_from_path(img_path)
        elapsed = time.monotonic() - start_time
        remaining = TIME_BUDGET - elapsed
        images_left = total - idx

        # If running low on time, skip ensemble and use first model only
        per_image_budget = remaining / max(images_left, 1)
        use_all_models = per_image_budget > 2 * len(sessions)

        pil_img = Image.open(str(img_path)).convert("RGB")
        orig_w, orig_h = pil_img.size
        img_array = np.array(pil_img)

        blob, scale, pad_w, pad_h = letterbox(img_array, INPUT_SIZE)
        input_tensor = blob[np.newaxis, ...]

        wbf_boxes_list = []
        wbf_scores_list = []
        wbf_labels_list = []

        models_to_run = sessions if use_all_models else sessions[:1]

        for session, input_name in models_to_run:
            outputs = session.run(None, {input_name: input_tensor})
            boxes_xyxy, scores, labels = decode_yolo_output(
                outputs[0], scale, pad_w, pad_h, orig_w, orig_h
            )

            if len(scores) == 0:
                continue

            norm_boxes = normalize_boxes(boxes_xyxy, orig_w, orig_h)
            wbf_boxes_list.append(norm_boxes)
            wbf_scores_list.append(scores)
            wbf_labels_list.append(labels)

        if not wbf_boxes_list:
            continue

        # If only one model produced results, skip WBF
        if len(wbf_boxes_list) == 1:
            merged_boxes = wbf_boxes_list[0]
            merged_scores = wbf_scores_list[0]
            merged_labels = wbf_labels_list[0]
        else:
            merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
                wbf_boxes_list, wbf_scores_list, wbf_labels_list,
                iou_thr=WBF_IOU_THRESH, skip_box_thr=WBF_SKIP_THRESH,
            )

        # Denormalize back to pixels
        merged_boxes[:, [0, 2]] *= orig_w
        merged_boxes[:, [1, 3]] *= orig_h

        # Limit detections
        if len(merged_scores) > MAX_DET:
            top_idx = np.argsort(merged_scores)[::-1][:MAX_DET]
            merged_boxes = merged_boxes[top_idx]
            merged_scores = merged_scores[top_idx]
            merged_labels = merged_labels[top_idx]

        for i in range(len(merged_scores)):
            x1, y1, x2, y2 = merged_boxes[i]
            bw = x2 - x1
            bh = y2 - y1
            if bw < 1 or bh < 1:
                continue
            all_predictions.append({
                "image_id": image_id,
                "category_id": 0 if DETECTION_ONLY else int(merged_labels[i]),
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(bw), 2), round(float(bh), 2)],
                "score": round(float(merged_scores[i]), 5),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(all_predictions, f)


if __name__ == "__main__":
    main()
