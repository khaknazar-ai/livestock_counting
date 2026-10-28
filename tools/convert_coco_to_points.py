"""
tools/convert_coco_to_points.py

Convert COCO JSON (bounding box) annotations to point annotations.

Most public livestock datasets (Roboflow, Kaggle) provide bounding boxes.
For density estimation we need point annotations (head/center of each animal).
This script takes the center of each bbox as the point annotation.

Input:  COCO JSON with bounding box annotations
Output: Per-image JSON files in project format:
        {"image_file": "xxx.jpg", "points": [[x,y], ...], "count": N}

Usage:
    python tools/convert_coco_to_points.py \
        --coco_json path/to/_annotations.coco.json \
        --img_dir path/to/images \
        --out_dir data/livestock/annotations \
        --copy_images data/livestock/images
"""

import os
import json
import shutil
import argparse
from tqdm import tqdm


def convert(coco_path, img_src_dir, ann_out_dir, img_out_dir=None,
            min_count=5):
    with open(coco_path, 'r') as f:
        coco = json.load(f)

    # Build lookup: image_id -> image info
    id_to_img = {img['id']: img for img in coco['images']}

    # Group annotations by image_id
    img_anns = {}
    for ann in coco['annotations']:
        iid = ann['image_id']
        if iid not in img_anns:
            img_anns[iid] = []
        # bbox is [x, y, width, height] in COCO format
        x, y, w, h = ann['bbox']
        cx = x + w / 2.0
        cy = y + h / 2.0
        img_anns[iid].append([round(cx), round(cy)])

    os.makedirs(ann_out_dir, exist_ok=True)
    if img_out_dir:
        os.makedirs(img_out_dir, exist_ok=True)

    kept = 0
    skipped = 0

    for img_id, img_info in tqdm(id_to_img.items(), desc="Converting"):
        filename = img_info['file_name']
        points = img_anns.get(img_id, [])

        # Skip images with too few animals (not useful for density estimation)
        if len(points) < min_count:
            skipped += 1
            continue

        stem = os.path.splitext(filename)[0]

        # Save annotation
        ann = {
            'image_file': filename,
            'points': points,
            'count': len(points)
        }
        ann_path = os.path.join(ann_out_dir, stem + '.json')
        with open(ann_path, 'w') as f:
            json.dump(ann, f, indent=2)

        # Copy image if requested
        if img_out_dir and img_src_dir:
            src = os.path.join(img_src_dir, filename)
            dst = os.path.join(img_out_dir, filename)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

        kept += 1

    print(f"\nDone: {kept} images kept, {skipped} skipped (< {min_count} animals)")
    print(f"Annotations: {ann_out_dir}")
    if img_out_dir:
        print(f"Images copied to: {img_out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_json', required=True,
                        help='Path to COCO JSON annotations file')
    parser.add_argument('--img_dir', required=True,
                        help='Directory containing the images')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for point annotation JSONs')
    parser.add_argument('--copy_images', default=None,
                        help='If set, copy images to this directory')
    parser.add_argument('--min_count', type=int, default=5,
                        help='Skip images with fewer than N animals')
    args = parser.parse_args()

    convert(args.coco_json, args.img_dir, args.out_dir,
            args.copy_images, args.min_count)


if __name__ == '__main__':
    main()
