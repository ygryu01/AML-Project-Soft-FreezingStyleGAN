#!/usr/bin/env python
"""sr256 FID measurements:
   - baseline normal (Hb(S0))
   - variant_a normal (Hv(Sv))
   - variant_a counterfactual (Hv(S0)) via stem swap
"""
import os, sys, json, copy, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

import torch
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
SUBDIR = '00000-stylegan3-t-imagenet100_256-gpus1-batch32'
DATA = os.path.join(REPO, 'data/imagenet100_256.zip')

BASELINE_DIR = 'imagenet100_256_converged_baseline'
VARIANT_DIR  = 'imagenet100_256_converged_variant_a'
OUT_BASE     = os.path.join(REPO, 'counterfactual_runs')
os.makedirs(OUT_BASE, exist_ok=True)


def load_G(path):
    with open(path, 'rb') as f:
        return legacy.load_network_pkl(f)['G_ema'].eval().to(device).requires_grad_(False)


def build_counterfactual_G(G_var, G_base):
    G_cf = copy.deepcopy(G_var).eval().to(device).requires_grad_(False)
    head_set = set(getattr(G_cf, 'head_layer_names', []) or [])
    G_cf.mapping = copy.deepcopy(G_base.mapping).eval().to(device).requires_grad_(False)
    n = 0
    for name in G_cf.synthesis.layer_names:
        if name in head_set or not hasattr(G_base.synthesis, name):
            continue
        setattr(G_cf.synthesis, name,
                copy.deepcopy(getattr(G_base.synthesis, name)).eval().to(device).requires_grad_(False))
        n += 1
    print(f'   swapped {n} stem layers + mapping')
    return G_cf


def run_metric(G, out_dir, snap_label):
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, 'metric-fid50k_full.jsonl')
    if os.path.isfile(jp):
        print(f'   skip — already done {jp}')
        return
    print(f'   running fid50k_full ...', flush=True)
    res = metric_main.calc_metric(
        metric='fid50k_full',
        G=G,
        dataset_kwargs=dnnlib.EasyDict(
            class_name='training.dataset.ImageFolderDataset',
            path=DATA, use_labels=True, xflip=True,
            force_label_dim=1000, resolution=256,
        ),
        num_gpus=1, rank=0, device=device, cache=True,
    )
    fid = float(res.results['fid50k_full'])
    print(f'   FID = {fid:.4f}  ({res.total_time_str})', flush=True)
    rec = dict(
        results=dict(res.results), metric='fid50k_full',
        total_time=res.total_time, total_time_str=res.total_time_str,
        num_gpus=1, snapshot_pkl=snap_label, timestamp=0,
    )
    with open(jp, 'w') as f:
        f.write(json.dumps(rec) + '\n')
    print(f'   wrote {jp}')


# Load baseline + variant
print('Loading baseline...')
G_base = load_G(os.path.join(RUNDIR, BASELINE_DIR, SUBDIR, 'network-snapshot.pkl'))
print(f'   layers={len(G_base.synthesis.layer_names)}')

# counterfactual only (user requested)
print('\n========== sr256 variant_a counterfactual Hv(S0) ==========')
G_var = load_G(os.path.join(RUNDIR, VARIANT_DIR, SUBDIR, 'network-snapshot.pkl'))
G_cf = build_counterfactual_G(G_var, G_base)
del G_var
run_metric(G_cf, os.path.join(OUT_BASE, 'imagenet100_256_variant_a_cf'), 'cf_sr256_a')
del G_cf
torch.cuda.empty_cache()

print('\nAll sr256 measurements done.')
