"""
tools/convert_labelme.py

Convert LabelMe point-annotation JSON files → project annotation format.

LabelMe exports look like:
  {
    "imagePath": "pen_001.jpg",
    "shapes": [
      {"label": "animal", "points": [[245.0, 130.0]], "shape_type": "point"},
      ...
    ]
  }

This script converts them to:
  {
    "image_file": "pen_001.jpg",
    "points": [[245, 130], [312, 98], ...],
    "count": 47
  }

Usage:
    python tools/convert_labelme.py \
        --src_dir path/to/labelme_jsons \
        --out_dir data/livestock/annotations
"""

import os
import json
import argparse
from tqdm import tqdm


def convert_single(labelme_path: str, out_path: str):
    with open(labelme_path, 'r', encoding='utf-8') as f:
        lm = json.load(f)

    image_file = os.path.basename(lm.get('imagePath', ''))
    points = []

    for shape in lm.get('shapes', []):
        if shape.get('shape_type') == 'point':
            xy = shape['points'][0]   # [[x, y]] for point type
            points.append([round(xy[0]), round(xy[1])])
        elif shape.get('shape_type') == 'rectangle':
            # Convert bounding box center to point
            pts = shape['points']   # [[x1,y1],[x2,y2]]
            cx = (pts[0][0] + pts[1][0]) / 2
            cy = (pts[0][1] + pts[1][1]) / 2
            points.append([round(cx), round(cy)])

    out = {
        'image_file': image_file,
        'points':     points,
        'count':      len(points)
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True, help='Directory with LabelMe JSONs')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    args = parser.parse_args()

    jsons = [f for f in os.listdir(args.src_dir) if f.endswith('.json')]
    print(f"Converting {len(jsons)} LabelMe files...")

    for fname in tqdm(jsons):
        src = os.path.join(args.src_dir, fname)
        out = os.path.join(args.out_dir, fname)
        convert_single(src, out)

    print(f"✓ Done. Converted annotations saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
