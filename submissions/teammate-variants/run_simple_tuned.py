import argparse
import json
from pathlib import Path

import torch

# Patch torch.load for torch 2.6+ / ultralytics 8.1.0 compatibility
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    model = YOLO("best.pt")
    predictions = []

    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])
        try:
            results = model(
                str(img_path),
                device=device,
                imgsz=1280,
                conf=0.10,
                half=use_cuda,
                verbose=False,
            )
            for r in results:
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(r.boxes.cls[i].item()),
                        "bbox": [
                            round(x1, 1),
                            round(y1, 1),
                            round(x2 - x1, 1),
                            round(y2 - y1, 1),
                        ],
                        "score": round(float(r.boxes.conf[i].item()), 4),
                    })
        except Exception:
            continue

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()
