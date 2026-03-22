"""
Visualize YOLO labels on images.

Usage:
    python scripts/visualize_labels.py --data-dir data/combined_data --num 5 --type prod
    python scripts/visualize_labels.py --data-dir data/combined_data --num 5 --type shelf
    python scripts/visualize_labels.py --data-dir data/combined_data --num 5  # random mix
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random


def draw_yolo_boxes(img_path, label_path, classes):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    w, h = img.size

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cat_id = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Convert YOLO normalized to pixel coords
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            label = classes[cat_id] if cat_id < len(classes) else str(cat_id)
            # Truncate long names
            label = label[:30]

            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1 + 2, y1 + 2), label, fill="red")

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--type", choices=["prod", "shelf", "all"], default="all")
    parser.add_argument("--output-dir", default="data/visualizations")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load class names
    classes_file = data_dir / "classes.txt"
    classes = []
    if classes_file.exists():
        classes = [line.strip() for line in open(classes_file)]

    # Get image list
    images = sorted((data_dir / "images").iterdir())
    if args.type == "prod":
        images = [i for i in images if i.stem.startswith("prod_")]
    elif args.type == "shelf":
        images = [i for i in images if not i.stem.startswith("prod_")]

    random.seed(42)
    samples = random.sample(images, min(args.num, len(images)))

    for img_path in samples:
        label_path = data_dir / "labels" / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"No label for {img_path.name}, skipping")
            continue

        result = draw_yolo_boxes(img_path, label_path, classes)
        out_path = output_dir / f"viz_{img_path.stem}.jpg"
        result.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
