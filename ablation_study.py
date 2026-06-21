#!/usr/bin/env python
"""Complete-stem ablation study with images and metrics.

Build ``H_variant(S_baseline)`` by keeping the variant head while replacing its
mapping, synthesis input, and every non-head synthesis layer with the baseline
counterparts.  A single invocation saves a qualitative comparison image and
computes the requested dataset metrics.

Examples:
    python ablation_study.py sr128_conv_a3
    python ablation_study.py sr128_conv_a3 --metrics fid50k_full,kid50k_full
    python ablation_study.py all --metrics fid50k_full
"""

import argparse
import copy
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import dnnlib  # noqa: E402
import legacy  # noqa: E402
from metrics import metric_main  # noqa: E402


class _MissingPickleClass:
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


@dataclass(frozen=True)
class Experiment:
    baseline_run: str
    variant_run: str
    data_zip: str
    resolution: int
    use_labels: bool = True


EXPERIMENTS = {
    # ImageNet-100, converged stem, 32 -> 64.
    "sr64_conv_a": Experiment("imagenet100_64_conv_baseline", "imagenet100_64_conv_a", "imagenet100_64.zip", 64),
    "sr64_conv_a2": Experiment("imagenet100_64_conv_baseline", "imagenet100_64_conv_a2", "imagenet100_64.zip", 64),
    "sr64_conv_a3": Experiment("imagenet100_64_conv_baseline", "imagenet100_64_conv_a3", "imagenet100_64.zip", 64),
    "sr64_conv_b": Experiment("imagenet100_64_conv_baseline", "imagenet100_64_conv_b", "imagenet100_64.zip", 64),
    "sr64_conv_c": Experiment("imagenet100_64_conv_baseline", "imagenet100_64_conv_c", "imagenet100_64.zip", 64),
    # ImageNet-100, converged stem, 64 -> 128.
    "sr128_conv_a": Experiment("imagenet100_128_conv_baseline", "imagenet100_128_conv_a", "imagenet100_128.zip", 128),
    "sr128_conv_a2": Experiment("imagenet100_128_conv_baseline", "imagenet100_128_conv_a2", "imagenet100_128.zip", 128),
    "sr128_conv_a3": Experiment("imagenet100_128_conv_baseline", "imagenet100_128_conv_a3", "imagenet100_128.zip", 128),
    "sr128_conv_c": Experiment("imagenet100_128_conv_baseline", "imagenet100_128_conv_c", "imagenet100_128.zip", 128),
    "sr128_conv_d_cosine": Experiment("imagenet100_128_conv_baseline", "imagenet100_128_variant_d_cosine", "imagenet100_128.zip", 128),
    "sr128_conv_d_linear": Experiment("imagenet100_128_conv_baseline", "imagenet100_128_variant_d_linear", "imagenet100_128.zip", 128),
    # ImageNet-100, underfit/from-scratch stem, 64 -> 128.
    "sr128_scratch_a": Experiment("imagenet100_128_scratch_baseline", "imagenet100_128_scratch_variant_a", "imagenet100_128.zip", 128),
    "sr128_scratch_a2": Experiment("imagenet100_128_scratch_baseline", "imagenet100_128_scratch_variant_a2", "imagenet100_128.zip", 128),
    "sr128_scratch_a3": Experiment("imagenet100_128_scratch_baseline", "imagenet100_128_scratch_variant_a3", "imagenet100_128.zip", 128),
    "sr128_scratch_b": Experiment("imagenet100_128_scratch_baseline", "imagenet100_128_scratch_variant_b", "imagenet100_128.zip", 128),
    "sr128_scratch_c": Experiment("imagenet100_128_scratch_baseline", "imagenet100_128_scratch_variant_c", "imagenet100_128.zip", 128),
    # ImageNet-100, converged stem, 128 -> 256.
    "sr256_conv_a": Experiment("imagenet100_256_converged_baseline", "imagenet100_256_converged_variant_a", "imagenet100_256.zip", 256),
    "sr256_conv_a2": Experiment("imagenet100_256_converged_baseline", "imagenet100_256_converged_variant_a2", "imagenet100_256.zip", 256),
    "sr256_conv_a3": Experiment("imagenet100_256_converged_baseline", "imagenet100_256_converged_variant_a3", "imagenet100_256.zip", 256),
    # Unconditional Pokemon 64 -> 128.
    "pokemon128_a": Experiment("pokemon128_test100", "pokemon128_variant_a", "pokemon128.zip", 128, False),
    "pokemon128_a2": Experiment("pokemon128_test100", "pokemon128_variant_a2", "pokemon128.zip", 128, False),
    "pokemon128_a3": Experiment("pokemon128_test100", "pokemon128_variant_a3", "pokemon128.zip", 128, False),
    "pokemon128_b": Experiment("pokemon128_test100", "pokemon128_variant_b", "pokemon128.zip", 128, False),
    "pokemon128_c": Experiment("pokemon128_test100", "pokemon128_variant_c", "pokemon128.zip", 128, False),
    "pokemon128_d_cosine": Experiment("pokemon128_test100", "pokemon128_variant_d_cosine", "pokemon128.zip", 128, False),
    "pokemon128_d_linear": Experiment("pokemon128_test100", "pokemon128_variant_d_linear", "pokemon128.zip", 128, False),
}


def find_snapshot(runs_dir, run_name):
    run_dir = runs_dir / run_name
    candidates = sorted(run_dir.glob("*/network-snapshot.pkl"))
    if not candidates:
        raise FileNotFoundError(f"no network-snapshot.pkl under {run_dir}")
    return candidates[0]


def load_generator(snapshot, device):
    with snapshot.open("rb") as file:
        generator = legacy.load_network_pkl(file)["G_ema"]
    return generator.eval().to(device).requires_grad_(False)


def frozen_copy(module, device):
    return copy.deepcopy(module).eval().to(device).requires_grad_(False)


def build_ablated_generator(variant, baseline, device):
    """Return H_variant(S_baseline), replacing the complete stem and mapping."""
    ablated = frozen_copy(variant, device)
    head_names = set(getattr(ablated, "head_layer_names", []) or [])
    ablated.mapping = frozen_copy(baseline.mapping, device)

    input_swapped = False
    if hasattr(ablated.synthesis, "input") and hasattr(baseline.synthesis, "input"):
        ablated.synthesis.input = frozen_copy(baseline.synthesis.input, device)
        input_swapped = True

    swapped_layers = []
    for name in ablated.synthesis.layer_names:
        if name in head_names:
            continue
        if not hasattr(baseline.synthesis, name):
            raise AttributeError(f"baseline is missing stem layer {name}")
        setattr(ablated.synthesis, name, frozen_copy(getattr(baseline.synthesis, name), device))
        swapped_layers.append(name)

    if not head_names:
        raise RuntimeError("variant has no head_layer_names; refusing to replace an ambiguous stem")
    print(
        f"ablation: baseline mapping + input={input_swapped} + "
        f"{len(swapped_layers)} stem layers; retained {len(head_names)} variant head layers"
    )
    return ablated, input_swapped, swapped_layers, sorted(head_names)


def make_inputs(generator, seed, class_idx, device):
    torch.manual_seed(seed)
    z = torch.randn(1, generator.z_dim, device=device)
    c = torch.zeros(1, generator.c_dim, device=device)
    if generator.c_dim:
        if not 0 <= class_idx < generator.c_dim:
            raise ValueError(f"class index {class_idx} is outside [0, {generator.c_dim})")
        c[0, class_idx] = 1
    return z, c


def generate(generator, z, c):
    with torch.no_grad():
        return generator(z, c, noise_mode="const").detach().cpu()


def display(image):
    array = image[0].permute(1, 2, 0).numpy()
    return np.clip((array + 1) / 2, 0, 1)


def save_comparison(label, baseline, variant, ablated, args, output_dir):
    rows = []
    for seed in args.seeds:
        z, c = make_inputs(baseline, seed, args.class_idx, args.device)
        base_image = display(generate(baseline, z, c))
        variant_image = display(generate(variant, z, c))
        ablated_image = display(generate(ablated, z, c))
        difference = np.abs(variant_image - ablated_image).mean(axis=2)
        rows.append((seed, base_image, variant_image, ablated_image, difference))

    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 3 * len(rows)), squeeze=False)
    titles = ("baseline H_b(S_0)", "variant H_v(S_v)", "ablation H_v(S_0)", "|variant - ablation|")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=10)
    for row, (seed, base_image, variant_image, ablated_image, difference) in enumerate(rows):
        axes[row, 0].imshow(base_image)
        axes[row, 1].imshow(variant_image)
        axes[row, 2].imshow(ablated_image)
        axes[row, 3].imshow(difference, cmap="magma", vmin=0, vmax=max(1e-8, difference.max()))
        axes[row, 0].set_ylabel(f"seed {seed}")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Complete-stem ablation study — {label}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    image_path = output_dir / "comparison.png"
    fig.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", image_path)
    return image_path


def calculate_metric(metric, generator, experiment, data_path, label, output_dir, args):
    output_path = output_dir / f"metric-{metric}.jsonl"
    if output_path.exists() and not args.force:
        print(f"metric exists, skipping: {output_path}")
        return json.loads(output_path.read_text().splitlines()[0])
    result = metric_main.calc_metric(
        metric=metric,
        G=generator,
        dataset_kwargs=dnnlib.EasyDict(
            class_name="training.dataset.ImageFolderDataset",
            path=str(data_path),
            use_labels=experiment.use_labels,
            xflip=True,
            force_label_dim=1000 if experiment.use_labels else 0,
            resolution=experiment.resolution,
        ),
        num_gpus=1,
        rank=0,
        device=args.device,
        cache=True,
    )
    record = {
        "results": dict(result.results),
        "metric": metric,
        "total_time": result.total_time,
        "total_time_str": result.total_time_str,
        "num_gpus": 1,
        "snapshot_pkl": f"ablation_study_{label}",
        "baseline_run": experiment.baseline_run,
        "variant_run": experiment.variant_run,
        "ablation": "H_variant(S_baseline)",
    }
    output_path.write_text(json.dumps(record) + "\n")
    print(f"{metric}: {dict(result.results)}")
    print("wrote", output_path)
    return record


def run_experiment(label, experiment, args):
    print(f"\n========== {label} ==========")
    baseline_path = find_snapshot(args.runs_dir, experiment.baseline_run)
    variant_path = find_snapshot(args.runs_dir, experiment.variant_run)
    data_path = args.data_dir / experiment.data_zip
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    output_dir = args.output_dir / label
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_generator(baseline_path, args.device)
    variant = load_generator(variant_path, args.device)
    ablated, input_swapped, stem_layers, head_layers = build_ablated_generator(
        variant, baseline, args.device
    )

    image_path = save_comparison(label, baseline, variant, ablated, args, output_dir)
    metric_records = {}
    for metric in args.metrics:
        record = calculate_metric(metric, ablated, experiment, data_path, label, output_dir, args)
        metric_records[metric] = record["results"]

    summary = {
        "experiment": label,
        "baseline_snapshot": str(baseline_path),
        "variant_snapshot": str(variant_path),
        "ablation": "H_variant(S_baseline)",
        "swapped": {
            "mapping": True,
            "synthesis_input": input_swapped,
            "stem_layers": stem_layers,
        },
        "retained_variant_head_layers": head_layers,
        "comparison_image": str(image_path),
        "metrics": metric_records,
    }
    summary_path = output_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", summary_path)

    del baseline, variant, ablated
    gc.collect()
    if args.device.type == "cuda":
        torch.cuda.empty_cache()


def comma_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def int_list(value):
    return [int(item) for item in comma_list(value)]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=[*EXPERIMENTS, "all"], help="experiment preset")
    parser.add_argument("--metrics", type=comma_list, default=["fid50k_full"], help="comma-separated metrics")
    parser.add_argument("--seeds", type=int_list, default=[0, 1, 2], help="comma-separated image seeds")
    parser.add_argument("--class-idx", type=int, default=209, help="class used for qualitative images")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "training-runs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ablation_study_runs")
    parser.add_argument("--force", action="store_true", help="overwrite existing metric JSONL files")
    return parser


def main():
    args = build_parser().parse_args()
    args.device = torch.device(args.device)
    labels = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    failures = {}
    for label in labels:
        try:
            run_experiment(label, EXPERIMENTS[label], args)
        except (FileNotFoundError, AttributeError, RuntimeError, ValueError) as error:
            if args.experiment != "all":
                raise
            failures[label] = str(error)
            print(f"SKIP {label}: {error}")
    if failures:
        print("\nSkipped experiments:")
        for label, reason in failures.items():
            print(f"  {label}: {reason}")


if __name__ == "__main__":
    main()
