import argparse
import glob
import io
import json
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageOps
from tqdm import tqdm


# Original ImageNet-1k label IDs used by this project's ImageNet-100 subset.
# Keeping the original IDs is intentional: training uses --label_dim=1000 and
# remains compatible with the official 1000-class StyleGAN-XL stems.
IMAGENET100_LABELS = {
    1, 14, 41, 63, 64, 75, 82, 93, 95, 97, 101, 103, 142, 143, 145, 150,
    195, 209, 223, 227, 244, 249, 256, 265, 266, 288, 310, 317, 323, 327,
    333, 338, 341, 362, 366, 394, 408, 414, 430, 444, 453, 458, 483, 488,
    497, 505, 516, 520, 523, 533, 545, 556, 561, 565, 573, 581, 597, 616,
    625, 626, 633, 640, 655, 684, 700, 720, 722, 727, 736, 747, 773, 776,
    802, 803, 818, 822, 824, 829, 844, 847, 849, 860, 864, 870, 886, 888,
    891, 896, 909, 911, 913, 920, 923, 929, 931, 934, 940, 960, 984, 988,
}


def get_image_bytes(x):
    if isinstance(x, dict):
        if x.get("bytes") is not None:
            return bytes(x["bytes"])
        if x.get("path") is not None:
            return Path(x["path"]).read_bytes()

    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)

    raise TypeError(f"Unsupported image field type: {type(x)}")


def center_crop_resize(img, resolution):
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    w, h = img.size
    crop = min(w, h)
    left = (w - crop) // 2
    top = (h - crop) // 2

    img = img.crop((left, top, left + crop, top + crop))
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.ANTIALIAS

    img = img.resize((resolution, resolution), resample)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_glob", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--image_col", default="image")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--no_labels", action="store_true")
    parser.add_argument(
        "--imagenet100",
        action="store_true",
        help="keep only the project's 100 original ImageNet-1k label IDs",
    )
    args = parser.parse_args()

    parquet_files = sorted(glob.glob(args.parquet_glob, recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files matched: {args.parquet_glob}")

    Path(args.dest).parent.mkdir(parents=True, exist_ok=True)

    labels = []
    idx = 0

    with zipfile.ZipFile(args.dest, "w", compression=zipfile.ZIP_STORED) as zf:
        for parquet_path in parquet_files:
            pf = pq.ParquetFile(parquet_path)

            for batch in pf.iter_batches(batch_size=args.batch_size):
                data = batch.to_pydict()

                if args.image_col not in data:
                    raise KeyError(f"Image column not found: {args.image_col}. Columns: {list(data.keys())}")

                images = data[args.image_col]

                has_labels = (not args.no_labels) and (args.label_col in data)
                batch_labels = data[args.label_col] if has_labels else None
                if args.imagenet100 and not has_labels:
                    raise KeyError("--imagenet100 requires a label column")

                for i, image_field in enumerate(images):
                    if args.max_images is not None and idx >= args.max_images:
                        break

                    lab = int(batch_labels[i]) if has_labels and batch_labels[i] is not None else None
                    if args.imagenet100 and lab not in IMAGENET100_LABELS:
                        continue

                    img_bytes = get_image_bytes(image_field)
                    img = Image.open(io.BytesIO(img_bytes))
                    img = center_crop_resize(img, args.resolution)

                    idx_str = f"{idx:08d}"
                    archive_fname = f"{idx_str[:5]}/img{idx_str}.png"

                    buf = io.BytesIO()
                    img.save(buf, format="PNG", compress_level=0, optimize=False)
                    zf.writestr(archive_fname, buf.getvalue())

                    if has_labels:
                        if lab is None:
                            labels.append(None)
                        else:
                            labels.append([archive_fname, lab])
                    else:
                        labels.append(None)

                    idx += 1

                if args.max_images is not None and idx >= args.max_images:
                    break

            if args.max_images is not None and idx >= args.max_images:
                break

        metadata = {
            "labels": labels if all(x is not None for x in labels) else None
        }
        zf.writestr("dataset.json", json.dumps(metadata))

    print(f"saved images: {idx}")
    print(f"output: {args.dest}")
    print(f"labels included: {metadata['labels'] is not None}")


if __name__ == "__main__":
    main()
