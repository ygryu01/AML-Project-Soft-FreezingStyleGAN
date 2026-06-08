#!/usr/bin/env python
"""Counterfactual FID/KID/PR for sr128 variant a and c.

Builds G_cf by deepcopying variant G_ema and replacing every stem-layer
module (and mapping) with the corresponding baseline module. The resulting
generator computes H_v(S_0(z,c)) for every forward — exactly the
counterfactual we want.

For each of (a, c):
   - Build G_cf
   - calc_metric on fid50k_full, kid50k_full, pr50k3_full
   - Save jsonl in counterfactual_runs/imagenet100_128_conv_<v>_cf/...
"""
import os, sys, json, copy, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

import numpy as np, torch
import dnnlib, legacy
from metrics import metric_main

_orig_find_class = legacy._LegacyUnpickler.find_class
def _patched_find_class(self, module, name):
    try:
        return _orig_find_class(self, module, name)
    except (ModuleNotFoundError, ImportError, AttributeError):
        return _Stub
legacy._LegacyUnpickler.find_class = _patched_find_class

os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device = {device}')

REPO = '/home/ygryu/log1/stylegan-xl'
RUNDIR = os.path.join(REPO, 'training-runs')
SUBDIR = '00000-stylegan3-t-imagenet100_128-gpus1-batch32'
DATA = os.path.join(REPO, 'data/imagenet100_128.zip')

BASELINE_DIR = 'imagenet100_128_conv_baseline'
TARGETS = [
    ('a3', 'imagenet100_128_conv_a3'),
]
METRICS = ['fid50k_full']

OUT_BASE = os.path.join(REPO, 'counterfactual_runs')
os.makedirs(OUT_BASE, exist_ok=True)


def load_G(path):
    with open(path, 'rb') as f:
        return legacy.load_network_pkl(f)['G_ema'].eval().to(device).requires_grad_(False)


def build_counterfactual_G(G_variant, G_baseline):
    """deep-copy variant G; replace every stem-layer (and mapping) module
    with baseline counterpart. Returns the patched generator."""
    G_cf = copy.deepcopy(G_variant).eval().to(device).requires_grad_(False)
    head_set = set(getattr(G_cf, 'head_layer_names', []) or [])
    # mapping (should already be same but make explicit)
    G_cf.mapping = copy.deepcopy(G_baseline.mapping).eval().to(device).requires_grad_(False)
    # stem layers
    n_swap = 0
    for name in G_cf.synthesis.layer_names:
        if name in head_set:
            continue
        if not hasattr(G_baseline.synthesis, name):
            print(f'   warning: baseline missing layer {name}, skipping')
            continue
        src = copy.deepcopy(getattr(G_baseline.synthesis, name)).eval().to(device).requires_grad_(False)
        setattr(G_cf.synthesis, name, src)
        n_swap += 1
    print(f'   swapped {n_swap} stem layers + mapping')
    return G_cf


def write_jsonl(run_dir, result_dict):
    name = result_dict['metric']
    p = os.path.join(run_dir, f'metric-{name}.jsonl')
    with open(p, 'a') as f:
        f.write(json.dumps(result_dict) + '\n')
    print(f'   wrote {p}')


def run_one_metric(G, run_dir, metric_name):
    print(f'   {metric_name} ...', flush=True)
    res = metric_main.calc_metric(
        metric=metric_name,
        G=G,
        dataset_kwargs=dnnlib.EasyDict(
            class_name='training.dataset.ImageFolderDataset',
            path=DATA,
            use_labels=True,
            xflip=True,
            force_label_dim=1000,
            resolution=128,
        ),
        num_gpus=1, rank=0, device=device, cache=True,
    )
    print(f'   {metric_name} done: {res.results}')
    res_dict = dict(
        results=dict(res.results),
        metric=metric_name,
        total_time=res.total_time,
        total_time_str=res.total_time_str,
        num_gpus=1,
        snapshot_pkl='counterfactual.pkl',
        timestamp=0,
    )
    write_jsonl(run_dir, res_dict)
    return res


# Load baseline once
print(f'Loading baseline G_ema...')
G_base = load_G(os.path.join(RUNDIR, BASELINE_DIR, SUBDIR, 'network-snapshot.pkl'))
print(f'   layers={len(G_base.synthesis.layer_names)} '
      f'head={list(getattr(G_base, "head_layer_names", []) or [])}')

for vname, vdir in TARGETS:
    print(f'\n========== variant {vname} ==========')
    G_var = load_G(os.path.join(RUNDIR, vdir, SUBDIR, 'network-snapshot.pkl'))
    G_cf  = build_counterfactual_G(G_var, G_base)
    del G_var
    run_dir = os.path.join(OUT_BASE, f'imagenet100_128_conv_{vname}_cf')
    os.makedirs(run_dir, exist_ok=True)
    print(f'   run_dir = {run_dir}')
    for m in METRICS:
        run_one_metric(G_cf, run_dir, m)
    del G_cf
    torch.cuda.empty_cache()

print('\nAll done.')
