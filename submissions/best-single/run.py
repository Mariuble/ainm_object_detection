"""
Grocery shelf object detection inference using ONNX Runtime.
Runs a YOLO11 model exported to ONNX format.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_SIZE = 1280
CONF_THRESH = 0.05
IOU_THRESH = 0.5
MAX_DET = 500
PAD_VALUE = 114  # standard YOLO letterbox gray
DETECTION_ONLY = False  # Set True to use category_id=0 for all predictions (max 70% score)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def letterbox(img_array: np.ndarray, new_shape: int = INPUT_SIZE):
    """Resize image with unchanged aspect ratio using padding.

    Args:
        img_array: HWC uint8 RGB numpy array.
        new_shape: target square size.

    Returns:
        padded: float32 CHW array normalized to [0, 1].
        scale: scale factor applied.
        pad_w: padding added on left.
        pad_h: padding added on top.
    """
    h, w = img_array.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    # Resize via PIL for quality (lanczos)
    resized = np.array(
        Image.fromarray(img_array).resize((new_w, new_h), Image.LANCZOS)
    )

    # Create padded canvas
    pad_w = (new_shape - new_w) // 2
    pad_h = (new_shape - new_h) // 2
    canvas = np.full((new_shape, new_shape, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w, :] = resized

    # HWC uint8 -> CHW float32 [0,1]
    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # CHW
    return blob, scale, pad_w, pad_h


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert cx,cy,w,h to x1,y1,x2,y2."""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return out


def nms_numpy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Pure-numpy NMS. Returns indices to keep."""
    if len(boxes_xyxy) == 0:
        return np.empty(0, dtype=np.int64)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def postprocess(output: np.ndarray, scale: float, pad_w: int, pad_h: int,
                orig_w: int, orig_h: int):
    """Post-process raw YOLO ONNX output into detections.

    Args:
        output: raw model output, shape (1, 4+nc, num_det) or (1, num_det, 4+nc).
        scale, pad_w, pad_h: letterbox transform params.
        orig_w, orig_h: original image dimensions.

    Returns:
        List of dicts with category_id, bbox [x,y,w,h], score.
    """
    pred = output[0]  # remove batch dim -> (4+nc, num_det) or (num_det, 4+nc)

    # Determine orientation: if dim0 is small (like 360) it's (4+nc, num_det)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T  # -> (num_det, 4+nc)

    # Split boxes and class scores
    boxes_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:]

    # Best class per detection
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]

    # Confidence filter
    mask = confidences >= CONF_THRESH
    boxes_cxcywh = boxes_cxcywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(boxes_cxcywh) == 0:
        return []

    # Convert to xyxy for NMS
    boxes_xyxy = xywh_to_xyxy(boxes_cxcywh)

    # Class-agnostic NMS (better for dense shelves with overlapping categories)
    keep = nms_numpy(boxes_xyxy, confidences, IOU_THRESH)
    if len(keep) > MAX_DET:
        keep = keep[:MAX_DET]

    boxes_xyxy = boxes_xyxy[keep]
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    # Map coordinates back to original image space
    # Remove padding offset, then undo scale
    boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_w) / scale
    boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_h) / scale
    boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_w) / scale
    boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_h) / scale

    # Clip to image bounds
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    # Convert xyxy -> COCO xywh
    results = []
    for idx in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[idx]
        bw = x2 - x1
        bh = y2 - y1
        if bw < 1 or bh < 1:
            continue
        results.append({
            "category_id": 0 if DETECTION_ONLY else int(class_ids[idx]),
            "bbox": [round(float(x1), 2), round(float(y1), 2),
                     round(float(bw), 2), round(float(bh), 2)],
            "score": round(float(confidences[idx]), 5),
        })

    return results


def image_id_from_path(p: Path) -> int:
    """Extract numeric image id from filename like img_00042.jpg -> 42."""
    stem = p.stem  # e.g. "img_00042"
    num_part = stem.split("_")[-1]
    return int(num_part)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--output", type=str, required=True,
                        help="Path for output predictions JSON")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    # Locate ONNX model (same directory as this script)
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "best.onnx"

    # Set up ONNX Runtime session with GPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(model_path), sess_options, providers=providers)

    input_name = session.get_inputs()[0].name

    # Gather image paths
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    all_predictions = []

    for img_path in image_paths:
        image_id = image_id_from_path(img_path)

        # Load image as RGB numpy array
        pil_img = Image.open(str(img_path)).convert("RGB")
        orig_w, orig_h = pil_img.size
        img_array = np.array(pil_img)

        # Letterbox
        blob, scale, pad_w, pad_h = letterbox(img_array, INPUT_SIZE)

        # Add batch dimension: (1, 3, 1280, 1280)
        input_tensor = blob[np.newaxis, ...]

        # Run inference
        outputs = session.run(None, {input_name: input_tensor})
        raw_output = outputs[0]  # (1, 4+nc, num_det) or (1, num_det, 4+nc)

        # Post-process
        dets = postprocess(raw_output, scale, pad_w, pad_h, orig_w, orig_h)

        for det in dets:
            det["image_id"] = image_id
            all_predictions.append(det)

    # Write output JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(all_predictions, f)


if __name__ == "__main__":
    main()
