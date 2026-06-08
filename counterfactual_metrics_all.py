#!/usr/bin/env python
"""Counterfactual FID across all available baseline + variant pairs.

For every experiment in EXPERIMENTS:
  build counterfactual G (variant ckpt with stem replaced by baseline stem)
  compute fid50k_full
  append to JSONL under counterfactual_runs/<run_label>/

Skips if metric jsonl already exists (idempotent restart).
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
OUT_BASE = os.path.join(REPO, 'counterfactual_runs')
os.makedirs(OUT_BASE, exist_ok=True)


def _find_snap(run_dir):
    if not os.path.isdir(run_dir):
        return None
    for sub in sorted(os.listdir(run_dir)):
        cand = os.path.join(run_dir, sub, 'network-snapshot.pkl')
        if os.path.isfile(cand):
            return cand
    return None


# (label, baseline_dir, variant_dir, data_zip_basename, resolution)
EXPERIMENTS = [
    # sr64 conv (5)
    ('sr64_conv_a',  'imagenet100_64_conv_baseline', 'imagenet100_64_conv_a',  'imagenet100_64.zip',  64),
    ('sr64_conv_a2', 'imagenet100_64_conv_baseline', 'imagenet100_64_conv_a2', 'imagenet100_64.zip',  64),
    ('sr64_conv_a3', 'imagenet100_64_conv_baseline', 'imagenet100_64_conv_a3', 'imagenet100_64.zip',  64),
    ('sr64_conv_b',  'imagenet100_64_conv_baseline', 'imagenet100_64_conv_b',  'imagenet100_64.zip',  64),
    ('sr64_conv_c',  'imagenet100_64_conv_baseline', 'imagenet100_64_conv_c',  'imagenet100_64.zip',  64),
    # sr128 conv (5; a3 already measured separately)
    ('sr128_conv_a',        'imagenet100_128_conv_baseline', 'imagenet100_128_conv_a',        'imagenet100_128.zip', 128),
    ('sr128_conv_a2',       'imagenet100_128_conv_baseline', 'imagenet100_128_conv_a2',       'imagenet100_128.zip', 128),
    ('sr128_conv_c',        'imagenet100_128_conv_baseline', 'imagenet100_128_conv_c',        'imagenet100_128.zip', 128),
    ('sr128_conv_d_cosine', 'imagenet100_128_conv_baseline', 'imagenet100_128_variant_d_cosine', 'imagenet100_128.zip', 128),
    ('sr128_conv_d_linear', 'imagenet100_128_conv_baseline', 'imagenet100_128_variant_d_linear', 'imagenet100_128.zip', 128),
    # sr128 scratch (5)
    ('sr128_scratch_a',  'imagenet100_128_scratch_baseline', 'imagenet100_128_scratch_variant_a',  'imagenet100_128.zip', 128),
    ('sr128_scratch_a2', 'imagenet100_128_scratch_baseline', 'imagenet100_128_scratch_variant_a2', 'imagenet100_128.zip', 128),
    ('sr128_scratch_a3', 'imagenet100_128_scratch_baseline', 'imagenet100_128_scratch_variant_a3', 'imagenet100_128.zip', 128),
    ('sr128_scratch_b',  'imagenet100_128_scratch_baseline', 'imagenet100_128_scratch_variant_b',  'imagenet100_128.zip', 128),
    ('sr128_scratch_c',  'imagenet100_128_scratch_baseline', 'imagenet100_128_scratch_variant_c',  'imagenet100_128.zip', 128),
]


def load_G(path):
    with open(path, 'rb') as f:
        return legacy.load_network_pkl(f)['G_ema'].eval().to(device).requires_grad_(False)


def build_counterfactual_G(G_var, G_base):
    G_cf = copy.deepcopy(G_var).eval().to(device).requires_grad_(False)
    head_set = set(getattr(G_cf, 'head_layer_names', []) or [])
    G_cf.mapping = copy.deepcopy(G_base.mapping).eval().to(device).requires_grad_(False)
    n = 0
    for name in G_cf.synthesis.layer_names:
        if name in head_set:
            continue
        if not hasattr(G_base.synthesis, name):
            continue
        setattr(G_cf.synthesis, name,
                copy.deepcopy(getattr(G_base.synthesis, name)).eval().to(device).requires_grad_(False))
        n += 1
    return G_cf, n


for i, (label, b_dir, v_dir, zip_name, res) in enumerate(EXPERIMENTS):
    print(f'\n========== [{i+1}/{len(EXPERIMENTS)}] {label} ==========', flush=True)
    out_dir = os.path.join(OUT_BASE, f'{label}_cf')
    out_jsonl = os.path.join(out_dir, 'metric-fid50k_full.jsonl')
    if os.path.isfile(out_jsonl):
        print(f'   already done — skip ({out_jsonl})')
        continue

    b_snap = _find_snap(os.path.join(RUNDIR, b_dir))
    v_snap = _find_snap(os.path.join(RUNDIR, v_dir))
    if b_snap is None or v_snap is None:
        print(f'   missing snapshot — baseline={b_snap} variant={v_snap}, skip')
        continue
    data_path = os.path.join(REPO, 'data', zip_name)
    if not os.path.isfile(data_path):
        print(f'   missing data zip {data_path}, skip')
        continue

    print(f'   loading baseline {b_dir}')
    G_b = load_G(b_snap)
    print(f'   loading variant  {v_dir}')
    G_v = load_G(v_snap)
    G_cf, n_swap = build_counterfactual_G(G_v, G_b)
    del G_v
    print(f'   swapped {n_swap} stem layers + mapping; res={res}')

    os.makedirs(out_dir, exist_ok=True)
    print(f'   running fid50k_full ...', flush=True)
    res_obj = metric_main.calc_metric(
        metric='fid50k_full',
        G=G_cf,
        dataset_kwargs=dnnlib.EasyDict(
            class_name='training.dataset.ImageFolderDataset',
            path=data_path, use_labels=True, xflip=True,
            force_label_dim=1000, resolution=res,
        ),
        num_gpus=1, rank=0, device=device, cache=True,
    )
    fid = float(res_obj.results['fid50k_full'])
    print(f'   FID = {fid:.4f}  ({res_obj.total_time_str})', flush=True)
    rec = dict(
        results=dict(res_obj.results),
        metric='fid50k_full',
        total_time=res_obj.total_time,
        total_time_str=res_obj.total_time_str,
        num_gpus=1, snapshot_pkl=f'cf_{label}',
        baseline_dir=b_dir, variant_dir=v_dir,
        timestamp=0,
    )
    with open(out_jsonl, 'w') as f:
        f.write(json.dumps(rec) + '\n')
    print(f'   wrote {out_jsonl}')
    del G_b, G_cf
    torch.cuda.empty_cache()

print('\nAll experiments done.')
