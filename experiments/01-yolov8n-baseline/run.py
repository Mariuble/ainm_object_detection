import argparse
import json
from pathlib import Path
import torch
from ultralytics import YOLO
 

def _enable_torch26_pt_compat():
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _enable_torch26_pt_compat()
    model = YOLO("yolov8n.pt")
    predictions = []
 
    for img in sorted(Path(args.input).iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img.stem.split("_")[-1])
        results = model(str(img), device=device, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(r.boxes.cls[i].item()),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                    "score": round(float(r.boxes.conf[i].item()), 3),
                })
 
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
 
if __name__ == "__main__":
    main()