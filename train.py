"""
Training script for YOLO11 grocery shelf object detection.
Replaces the Colab notebook workflow for local/GCP execution.

Usage:
    python vibe/train.py --data-dir data/colab_data --train-pct 0.95
"""

import argparse
import random
from pathlib import Path
import shutil
import yaml


def train_val_split(data_path: Path, train_pct: float):
    """Split images/labels into train and validation folders."""
    input_images = data_path / "images"
    input_labels = data_path / "labels"

    train_img = data_path / "split" / "train" / "images"
    train_lbl = data_path / "split" / "train" / "labels"
    val_img = data_path / "split" / "val" / "images"
    val_lbl = data_path / "split" / "val" / "labels"

    # Clear previous split
    split_dir = data_path / "split"
    if split_dir.exists():
        shutil.rmtree(split_dir)

    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in input_images.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    random.shuffle(img_files)

    train_count = int(len(img_files) * train_pct)
    train_files = img_files[:train_count]
    val_files = img_files[train_count:]

    for img_file in train_files:
        shutil.copy2(img_file, train_img / img_file.name)
        lbl_file = input_labels / (img_file.stem + ".txt")
        if lbl_file.exists():
            shutil.copy2(lbl_file, train_lbl / lbl_file.name)

    for img_file in val_files:
        shutil.copy2(img_file, val_img / img_file.name)
        lbl_file = input_labels / (img_file.stem + ".txt")
        if lbl_file.exists():
            shutil.copy2(lbl_file, val_lbl / lbl_file.name)

    print(f"Split: {len(train_files)} train, {len(val_files)} val")
    return data_path / "split"


def create_data_yaml(classes_txt: Path, split_dir: Path, output_yaml: Path):
    """Create data.yaml with correct nc and preserved class indices."""
    with open(classes_txt, "r") as f:
        classes = [line.rstrip("\n") for line in f.readlines()]

    # Use dict format to preserve empty class names (e.g. category 300)
    names_dict = {i: name for i, name in enumerate(classes)}

    data = {
        "path": str(split_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(classes),
        "names": names_dict,
    }

    with open(output_yaml, "w") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"Created {output_yaml} with nc={len(classes)}")
    return output_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to data directory (images/, labels/, classes.txt)")
    parser.add_argument("--train-pct", type=float, default=0.95)
    parser.add_argument("--model", default="yolo11l.pt")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--cls", type=float, default=0.5, help="Classification loss weight")
    parser.add_argument("--copy-paste", type=float, default=0.0, help="Copy-paste augmentation probability")
    parser.add_argument("--close-mosaic", type=int, default=20, help="Disable mosaic N epochs before end")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    classes_txt = data_dir / "classes.txt"

    if not classes_txt.exists():
        print(f"Error: {classes_txt} not found")
        return

    # Split data
    split_dir = train_val_split(data_dir, args.train_pct)

    # Create data.yaml
    yaml_path = data_dir / "data.yaml"
    create_data_yaml(classes_txt, split_dir, yaml_path)

    # Train
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        cos_lr=True,
        close_mosaic=args.close_mosaic,
        cls=args.cls,
        copy_paste=args.copy_paste,
        amp=True,
    )

    # Export best model to ONNX
    best_pt = Path("runs/detect/train/weights/best.pt")
    if best_pt.exists():
        best_model = YOLO(str(best_pt))
        best_model.export(format="onnx", imgsz=args.imgsz, opset=17)
        print(f"Exported ONNX to {best_pt.with_suffix('.onnx')}")


if __name__ == "__main__":
    main()
