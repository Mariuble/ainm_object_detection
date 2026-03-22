"""
Convert product reference images into YOLO training data.

Each product reference image (front, back, left, right, main, etc.) becomes
a training example with a full-image bounding box and the correct class label.

Usage:
    python scripts/add_product_images.py \
        --annotations data/train/annotations.json \
        --product-images data/NM_NGD_product_images \
        --output-dir data/colab_data_with_products

This creates an output directory with images/ and labels/ folders that include
both the original shelf data AND the new product reference images.
"""

import json
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="Path to COCO annotations.json")
    parser.add_argument("--product-images", required=True, help="Path to NM_NGD_product_images dir")
    parser.add_argument("--existing-data", default=None, help="Path to existing colab_data dir (images/ + labels/) to copy from")
    parser.add_argument("--output-dir", required=True, help="Output directory for combined dataset")
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    product_images_path = Path(args.product_images)
    output_dir = Path(args.output_dir)
    output_images = output_dir / "images"
    output_labels = output_dir / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    # Load COCO categories: name -> category_id
    with open(annotations_path) as f:
        coco = json.load(f)
    name_to_catid = {c["name"]: c["id"] for c in coco["categories"]}

    # Load product metadata
    metadata_path = product_images_path / "metadata.json"
    with open(metadata_path) as f:
        meta = json.load(f)

    # Copy existing shelf data if provided
    if args.existing_data:
        existing = Path(args.existing_data)
        print(f"Copying existing data from {existing}...")
        copied = 0
        for img_file in (existing / "images").iterdir():
            shutil.copy2(img_file, output_images / img_file.name)
            copied += 1
        for lbl_file in (existing / "labels").iterdir():
            shutil.copy2(lbl_file, output_labels / lbl_file.name)
        print(f"  Copied {copied} existing images + labels")

    # Process product reference images
    added = 0
    skipped = 0
    for product in meta["products"]:
        product_name = product["product_name"]
        product_code = product["product_code"]

        if product_name not in name_to_catid:
            print(f"  Skipping {product_name} - no matching category")
            skipped += 1
            continue

        if not product["has_images"]:
            skipped += 1
            continue

        cat_id = name_to_catid[product_name]
        product_dir = product_images_path / product_code

        if not product_dir.exists():
            skipped += 1
            continue

        for img_file in sorted(product_dir.iterdir()):
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            # Name: prod_{product_code}_{view}.jpg
            new_name = f"prod_{product_code}_{img_file.stem}"
            new_img_path = output_images / f"{new_name}{img_file.suffix}"
            new_lbl_path = output_labels / f"{new_name}.txt"

            # Copy image
            shutil.copy2(img_file, new_img_path)

            # Write YOLO label: full-image bounding box
            # class_id x_center y_center width height (all normalized)
            with open(new_lbl_path, "w") as f:
                f.write(f"{cat_id} 0.5 0.5 1.0 1.0\n")

            added += 1

    print(f"\nDone! Added {added} product reference images, skipped {skipped}")
    print(f"Output: {output_dir}")

    # Also copy classes.txt if it exists alongside annotations
    classes_txt = annotations_path.parent / "classes.txt"
    if classes_txt.exists():
        shutil.copy2(classes_txt, output_dir / "classes.txt")
        print(f"Copied classes.txt")

    # Print totals
    total_images = len(list(output_images.iterdir()))
    total_labels = len(list(output_labels.iterdir()))
    print(f"Total images: {total_images}, Total labels: {total_labels}")


if __name__ == "__main__":
    main()
