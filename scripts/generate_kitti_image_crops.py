"""Generate image crops from KITTI 2D boxes stored in frustum pickles."""

from __future__ import print_function

import os
import argparse

try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    from PIL import Image
except ImportError:
    Image = None


def load_pickle_boxes(pickle_path):
    with open(pickle_path, "rb") as fp:
        id_list = pickle.load(fp)
        box2d_list = pickle.load(fp)
        return id_list, box2d_list


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width - 1))
    y2 = max(0, min(int(round(y2)), height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pickle_path",
        required=True,
        help="Path to frustum pickle (e.g., kitti/frustum_carpedcyc_train.pickle)",
    )
    parser.add_argument(
        "--kitti_image_dir",
        default=os.path.join("dataset", "KITTI", "object", "training", "image_2"),
        help="KITTI image_2 directory",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for cropped images",
    )
    parser.add_argument(
        "--ext",
        default="png",
        help="Image extension (default: png)",
    )
    parser.add_argument(
        "--resize",
        default="224,224",
        help="Resize crops to WxH (default: 224,224)",
    )
    parser.add_argument(
        "--max_count",
        type=int,
        default=-1,
        help="Max number of crops to write (-1 for all)",
    )
    args = parser.parse_args()

    if Image is None:
        raise ImportError("PIL not found. Please install pillow.")

    id_list, box2d_list = load_pickle_boxes(args.pickle_path)
    os.makedirs(args.output_dir, exist_ok=True)

    resize_parts = args.resize.split(",")
    if len(resize_parts) != 2:
        raise ValueError("--resize must be in W,H format")
    resize_w = int(resize_parts[0])
    resize_h = int(resize_parts[1])

    written = 0
    for i, (img_id, box2d) in enumerate(zip(id_list, box2d_list)):
        if args.max_count >= 0 and written >= args.max_count:
            break

        image_name = "%06d.%s" % (img_id, args.ext)
        image_path = os.path.join(args.kitti_image_dir, image_name)
        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        box = clamp_box(box2d, width, height)
        if box is None:
            continue

        crop = image.crop(box)
        crop = crop.resize((resize_w, resize_h), Image.BILINEAR)
        crop_name = "%06d_%06d.png" % (img_id, i)
        crop.save(os.path.join(args.output_dir, crop_name))
        written += 1

    print("Wrote %d crops to %s" % (written, args.output_dir))


if __name__ == "__main__":
    main()
