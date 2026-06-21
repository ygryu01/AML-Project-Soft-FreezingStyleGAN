#!/usr/bin/env python
"""Unified Grad-CAM saliency analyses for the StyleGAN-XL dog experiments.

This file consolidates the former ``saliency_{image_all,layers_all,diff,avg,
objbg}_dogs.py`` scripts.  The shared snapshot loading, layer hooks, Grad-CAM
calculation, sample creation, and plotting helpers live here once.

Examples:
    python saliency_dogs.py image
    python saliency_dogs.py layers --class-idx 209 --seed 1
    python saliency_dogs.py diff --min-layer 9
    python saliency_dogs.py average --num-seeds 2
    python saliency_dogs.py objbg --num-seeds 2

All commands default to CPU and write into ``featuremaps/``.  ``objbg`` is the
only command that imports torchvision and may download segmentation weights.
"""

import argparse
import gc
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import dnnlib  # noqa: E402,F401 - required while unpickling snapshots
import legacy  # noqa: E402


class _MissingPickleClass:
    """Fallback for irrelevant classes unavailable in old snapshot modules."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


_original_find_class = legacy._LegacyUnpickler.find_class


def _safe_find_class(unpickler, module, name):
    try:
        return _original_find_class(unpickler, module, name)
    except (ModuleNotFoundError, ImportError, AttributeError):
        return _MissingPickleClass


legacy._LegacyUnpickler.find_class = _safe_find_class

SUBDIR = "00000-stylegan3-t-imagenet100_128-gpus1-batch32"
NETWORKS = [
    ("baseline", "imagenet100_128_conv_baseline"),
    ("a", "imagenet100_128_conv_a"),
    ("a2", "imagenet100_128_conv_a2"),
    ("a3", "imagenet100_128_conv_a3"),
    ("c", "imagenet100_128_conv_c"),
    ("d_cosine", "imagenet100_128_variant_d_cosine"),
    ("d_linear", "imagenet100_128_variant_d_linear"),
]
DOG_CLASSES = [195, 209, 223, 227, 244, 249, 256, 265, 266]
IMAGE_SIZE = 128


def load_generator(run_dir, args):
    snapshot = args.runs_dir / run_dir / args.snapshot_subdir / "network-snapshot.pkl"
    with snapshot.open("rb") as file:
        generator = legacy.load_network_pkl(file)["G_ema"]
    return generator.eval().to(args.device).requires_grad_(False)


def layer_number(name):
    try:
        return int(name[1:].split("_")[0])
    except (ValueError, IndexError):
        return -1


def get_layer_info(args, all_layers=False):
    generator = load_generator(NETWORKS[0][1], args)
    names = list(generator.synthesis.layer_names)
    head = set(getattr(generator, "head_layer_names", []) or [])
    del generator
    gc.collect()
    if not all_layers:
        names = [name for name in names if layer_number(name) >= args.min_layer]
    print("layers:", names)
    return names, head


def make_inputs(generator, seed, class_idx, device, requires_grad=True):
    torch.manual_seed(seed)
    z = torch.randn(1, generator.z_dim, device=device, requires_grad=requires_grad)
    c = torch.zeros(1, generator.c_dim, device=device)
    c[0, class_idx] = 1.0
    return z, c


def display_image(image):
    array = image[0].detach().permute(1, 2, 0).cpu().numpy()
    return np.clip((array + 1) / 2, 0, 1)


def compute_saliency(generator, z, c, layer_names):
    """Return generated image and normalized, upsampled Grad-CAM per layer."""
    features = {}
    handles = []
    for name in layer_names:
        def hook(_module, _inputs, output, layer=name):
            if torch.is_tensor(output) and output.dim() == 4:
                features[layer] = output

        handles.append(getattr(generator.synthesis, name).register_forward_hook(hook))

    try:
        image = generator(z, c, noise_mode="const")
    finally:
        for handle in handles:
            handle.remove()

    active_names = [name for name in layer_names if name in features]
    activations = [features[name] for name in active_names]
    gradients = torch.autograd.grad(
        image.square().sum(), activations, retain_graph=False, allow_unused=True
    )
    cams = {name: None for name in layer_names}
    for name, activation, gradient in zip(active_names, activations, gradients):
        if gradient is None:
            continue
        cam = torch.relu((activation * gradient).sum(1))[0]
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(
            cam[None, None], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
            align_corners=False,
        )[0, 0]
        cams[name] = cam.detach().cpu().numpy()
    return display_image(image), cams


def saliency_for_network(run_dir, args, layer_names, seed=None, class_idx=None):
    generator = load_generator(run_dir, args)
    result = saliency_for_generator(generator, args, layer_names, seed, class_idx)
    del generator
    gc.collect()
    return result


def saliency_for_generator(generator, args, layer_names, seed=None, class_idx=None):
    """Compute one sample while reusing an already loaded generator."""
    z, c = make_inputs(
        generator,
        args.seed if seed is None else seed,
        args.class_idx if class_idx is None else class_idx,
        args.device,
    )
    image, cams = compute_saliency(generator, z, c, layer_names)
    del z, c
    return image, cams


def clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def save_figure(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def run_image(args):
    layer_names, _ = get_layer_info(args, all_layers=True)
    data = {}
    for label, run_dir in NETWORKS:
        print("computing", label)
        image, cams = saliency_for_network(run_dir, args, layer_names)
        valid = [cam for cam in cams.values() if cam is not None]
        aggregate = np.mean(valid, axis=0)
        aggregate /= aggregate.max() + 1e-8
        data[label] = image, aggregate

    fig, axes = plt.subplots(2, len(NETWORKS), figsize=(2.3 * len(NETWORKS), 4.8))
    for column, (label, _) in enumerate(NETWORKS):
        image, saliency = data[label]
        axes[0, column].imshow(image)
        axes[0, column].set_title(label, fontsize=10)
        axes[1, column].imshow(image)
        axes[1, column].imshow(saliency, cmap="jet", alpha=0.5)
        clean_axes(axes[0, column])
        clean_axes(axes[1, column])
    axes[0, 0].set_ylabel("generated", fontsize=10)
    axes[1, 0].set_ylabel("saliency", fontsize=10)
    fig.suptitle(
        "Image-level gradient saliency (aggregate Grad-CAM over layers) — same seed\n"
        f"{args.pretty} (class {args.class_idx}, seed {args.seed}); baseline vs variants",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(
        fig, args.output_dir / f"saliency_image_all_dog_{args.slug}_s{args.seed:02d}.png", 140
    )


def run_layers(args):
    layer_names, head = get_layer_info(args)
    data = {}
    for label, run_dir in NETWORKS:
        print("computing", label)
        data[label] = saliency_for_network(run_dir, args, layer_names)

    fig, axes = plt.subplots(
        len(NETWORKS), 1 + len(layer_names),
        figsize=(2.1 * (1 + len(layer_names)), 2.1 * len(NETWORKS)),
        squeeze=False,
    )
    for row, (label, _) in enumerate(NETWORKS):
        image, cams = data[label]
        axes[row, 0].imshow(image)
        axes[row, 0].set_ylabel(label, fontsize=11)
        clean_axes(axes[row, 0])
        if row == 0:
            axes[row, 0].set_title("generated", fontsize=8)
        for column, name in enumerate(layer_names, 1):
            if cams[name] is not None:
                axes[row, column].imshow(cams[name], cmap="jet")
            clean_axes(axes[row, column])
            if row == 0:
                role = "head" if name in head else "stem"
                axes[row, column].set_title(f"{name}\n({role})", fontsize=6)
    fig.suptitle(
        "Per-layer gradient saliency (Grad-CAM, map only) — baseline vs variants\n"
        f"{args.pretty} (class {args.class_idx}, seed {args.seed})", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(
        fig,
        args.output_dir
        / f"saliency_layers_L{args.min_layer}toL26_maponly_all_dog_{args.slug}_s{args.seed:02d}.png",
        130,
    )


def run_diff(args):
    layer_names, _ = get_layer_info(args)
    print("baseline")
    _, baseline = saliency_for_network(NETWORKS[0][1], args, layer_names)
    variants = NETWORKS[1:]
    images, diffs = {}, {}
    for label, run_dir in variants:
        print("variant", label)
        image, cams = saliency_for_network(run_dir, args, layer_names)
        images[label] = image
        diffs[label] = {
            name: abs(cams[name] - baseline[name])
            if cams[name] is not None and baseline[name] is not None else None
            for name in layer_names
        }
    vmax = {
        name: max(
            [diffs[label][name].max() for label, _ in variants if diffs[label][name] is not None]
            or [1.0]
        ) + 1e-8
        for name in layer_names
    }
    fig, axes = plt.subplots(
        len(variants), 1 + len(layer_names),
        figsize=(1.9 * (1 + len(layer_names)), 2.0 * len(variants)), squeeze=False,
    )
    for row, (label, _) in enumerate(variants):
        axes[row, 0].imshow(images[label])
        axes[row, 0].set_ylabel(label, fontsize=11)
        clean_axes(axes[row, 0])
        if row == 0:
            axes[row, 0].set_title("generated", fontsize=8)
        for column, name in enumerate(layer_names, 1):
            diff = diffs[label][name]
            if diff is None:
                axes[row, column].set_visible(False)
                continue
            axes[row, column].imshow(diff, cmap="magma", vmin=0, vmax=vmax[name])
            title = f"{name.split('_')[0]}\nμ={diff.mean():.3f}" if row == 0 else f"μ={diff.mean():.3f}"
            axes[row, column].set_title(title, fontsize=6 if row == 0 else 5)
            clean_axes(axes[row, column])
    fig.suptitle(
        f"Per-layer saliency difference from baseline — {args.pretty} (seed {args.seed})\n"
        "brighter = larger |variant - baseline|; color scaled per layer",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(
        fig,
        args.output_dir
        / f"saliency_diff_L{args.min_layer}toL26_dog_{args.slug}_s{args.seed:02d}.png",
        130,
    )


def sample_pairs(num_seeds):
    return [(seed, class_idx) for seed in range(num_seeds) for class_idx in DOG_CLASSES]


def run_average(args):
    layer_names, head = get_layer_info(args)
    samples = sample_pairs(args.num_seeds)
    count = len(samples)
    print(f"{count} samples ({len(DOG_CLASSES)} breeds x {args.num_seeds} seeds)")
    data = {}
    for label, run_dir in NETWORKS:
        print("averaging", label)
        generator = load_generator(run_dir, args)
        image_sum = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float64)
        cam_sums = {name: np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64) for name in layer_names}
        for seed, class_idx in samples:
            image, cams = saliency_for_generator(generator, args, layer_names, seed, class_idx)
            image_sum += image
            for name, cam in cams.items():
                if cam is not None:
                    cam_sums[name] += cam
        averaged = {name: cam_sums[name] / count for name in layer_names}
        for name in layer_names:
            averaged[name] /= averaged[name].max() + 1e-8
        data[label] = image_sum / count, averaged
        del generator
        gc.collect()

    fig, axes = plt.subplots(
        len(NETWORKS), 1 + len(layer_names),
        figsize=(1.9 * (1 + len(layer_names)), 2.0 * len(NETWORKS)), squeeze=False,
    )
    for row, (label, _) in enumerate(NETWORKS):
        image, cams = data[label]
        axes[row, 0].imshow(np.clip(image, 0, 1))
        axes[row, 0].set_ylabel(label, fontsize=11)
        clean_axes(axes[row, 0])
        if row == 0:
            axes[row, 0].set_title(f"mean image\n({count} samples)", fontsize=7)
        for column, name in enumerate(layer_names, 1):
            axes[row, column].imshow(cams[name], cmap="jet")
            clean_axes(axes[row, column])
            if row == 0:
                role = "head" if name in head else "stem"
                axes[row, column].set_title(f"{name.split('_')[0]}\n({role})", fontsize=6)
    fig.suptitle(
        f"Averaged per-layer Grad-CAM over {count} dog samples — map only", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(
        fig,
        args.output_dir
        / f"saliency_avg_maponly_L{args.min_layer}toL26_dogs_{count}samples.png",
        130,
    )


def run_objbg(args):
    from torchvision.models.segmentation import (
        DeepLabV3_MobileNet_V3_Large_Weights as SegmentationWeights,
        deeplabv3_mobilenet_v3_large,
    )

    layer_names, head = get_layer_info(args)
    samples = sample_pairs(args.num_seeds)
    baseline = load_generator(NETWORKS[0][1], args)
    segmenter = deeplabv3_mobilenet_v3_large(
        weights=SegmentationWeights.DEFAULT
    ).eval().to(args.device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)
    masks = {}
    for seed, class_idx in samples:
        z, c = make_inputs(baseline, seed, class_idx, args.device, requires_grad=False)
        with torch.no_grad():
            image = ((baseline(z, c, noise_mode="const") + 1) / 2).clamp(0, 1)
            prediction = segmenter((image - mean) / std)["out"][0].argmax(0)
        mask = (prediction == 12).float().cpu().numpy()  # Pascal VOC dog
        if mask.mean() < 0.01:
            mask = (prediction != 0).float().cpu().numpy()
        masks[(seed, class_idx)] = mask
        print(f"mask seed={seed} class={class_idx} area={mask.mean():.3f}")
    valid = [sample for sample in samples if masks[sample].mean() >= 0.01]
    print(f"valid samples: {len(valid)}/{len(samples)}")
    del segmenter, baseline
    gc.collect()

    object_fraction = {label: {name: [] for name in layer_names} for label, _ in NETWORKS}
    enrichment = {label: {name: [] for name in layer_names} for label, _ in NETWORKS}
    for label, run_dir in NETWORKS:
        print("network", label)
        generator = load_generator(run_dir, args)
        for seed, class_idx in valid:
            _, cams = saliency_for_generator(generator, args, layer_names, seed, class_idx)
            mask = masks[(seed, class_idx)]
            area = mask.mean() + 1e-8
            for name, cam in cams.items():
                if cam is None:
                    continue
                fraction = float((cam * mask).sum() / (cam.sum() + 1e-8))
                object_fraction[label][name].append(fraction)
                enrichment[label][name].append(fraction / area)
        del generator
        gc.collect()

    xs = list(range(len(layer_names)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(13, 0.7 * len(layer_names)), 5.2))
    specs = [
        (ax1, object_fraction, "fraction of saliency on object", "Saliency mass on dog"),
        (ax2, enrichment, "object fraction / object area", "Object enrichment"),
    ]
    for ax, values, ylabel, title in specs:
        for index, name in enumerate(layer_names):
            if name in head:
                ax.axvspan(index - 0.5, index + 0.5, color="0.9", zorder=0)
        for label, _ in NETWORKS:
            ys = [np.mean(values[label][name]) if values[label][name] else np.nan for name in layer_names]
            ax.plot(xs, ys, marker="o", ms=3, lw=2.4 if label in ("baseline", "d_linear") else 1, label=label)
        ax.set_xticks(xs)
        ax.set_xticklabels(layer_names, rotation=90, fontsize=6)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    ax2.axhline(1.0, color="k", ls=":", lw=1)
    ax1.legend(fontsize=8, ncol=2)
    fig.suptitle(f"Object-vs-background saliency (average of {len(valid)} dog samples)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(
        fig,
        args.output_dir
        / f"saliency_objbg_L{args.min_layer}toL26_dogs_{len(valid)}samples.png",
        140,
    )
    print("mean enrichment over layers:")
    for label, _ in NETWORKS:
        values = [np.mean(enrichment[label][name]) for name in layer_names if enrichment[label][name]]
        print(f"  {label:10s} {np.mean(values):.3f}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("image", "layers", "diff", "average", "objbg"))
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "training-runs")
    parser.add_argument("--snapshot-subdir", default=SUBDIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "featuremaps")
    parser.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    parser.add_argument("--class-idx", type=int, default=209)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--slug", default="chesapeake_bay_retriever")
    parser.add_argument("--pretty", default="Chesapeake Bay retriever")
    parser.add_argument("--min-layer", type=int, default=9)
    parser.add_argument("--num-seeds", type=int, default=2)
    return parser


def main():
    args = build_parser().parse_args()
    args.device = torch.device(args.device)
    print("device =", args.device)
    handlers = {
        "image": run_image,
        "layers": run_layers,
        "diff": run_diff,
        "average": run_average,
        "objbg": run_objbg,
    }
    handlers[args.mode](args)
    print("done")


if __name__ == "__main__":
    main()
