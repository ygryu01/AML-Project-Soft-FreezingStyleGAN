#!/usr/bin/env python
"""Counterfactual FID for pokemon128 variant B vs baseline (test100, head-only).

Builds counterfactual G by replacing the stem of pokemon128_variant_b
with the stem from pokemon128_test100 (the head-only baseline whose
stem is the original underfit pokemon64_test100 stem).
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
SUBDIR = '00000-stylegan3-t-pokemon128-gpus1-batch32'
DATA = os.path.join(REPO, 'data/pokemon128.zip')

BASELINE_DIR = 'pokemon128_test100'
TARGETS = [
    ('b', 'pokemon128_variant_b'),
]
METRICS = ['fid50k_full']

OUT_BASE = os.path.join(REPO, 'counterfactual_runs')
os.makedirs(OUT_BASE, exist_ok=True)


def load_G(path):
    with open(path, 'rb') as f:
        return legacy.load_network_pkl(f)['G_ema'].eval().to(device).requires_grad_(False)


def build_counterfactual_G(G_variant, G_baseline):
    G_cf = copy.deepcopy(G_variant).eval().to(device).requires_grad_(False)
    head_set = set(getattr(G_cf, 'head_layer_names', []) or [])
    G_cf.mapping = copy.deepcopy(G_baseline.mapping).eval().to(device).requires_grad_(False)
    n = 0
    for name in G_cf.synthesis.layer_names:
        if name in head_set:
            continue
        if not hasattr(G_baseline.synthesis, name):
            continue
        setattr(G_cf.synthesis, name,
                copy.deepcopy(getattr(G_baseline.synthesis, name)).eval().to(device).requires_grad_(False))
        n += 1
    print(f'   swapped {n} stem layers + mapping')
    return G_cf


print(f'Loading baseline G_ema...')
G_base = load_G(os.path.join(RUNDIR, BASELINE_DIR, SUBDIR, 'network-snapshot.pkl'))
print(f'   layers={len(G_base.synthesis.layer_names)} head_layers={list(getattr(G_base, "head_layer_names", []) or [])}')

for vname, vdir in TARGETS:
    print(f'\n========== variant {vname} ==========')
    G_var = load_G(os.path.join(RUNDIR, vdir, SUBDIR, 'network-snapshot.pkl'))
    G_cf  = build_counterfactual_G(G_var, G_base)
    del G_var
    out_dir = os.path.join(OUT_BASE, f'pokemon128_{vname}_cf')
    os.makedirs(out_dir, exist_ok=True)
    print(f'   out_dir = {out_dir}')
    for m in METRICS:
        out_jsonl = os.path.join(out_dir, f'metric-{m}.jsonl')
        if os.path.isfile(out_jsonl):
            print(f'   skip — already done {out_jsonl}')
            continue
        print(f'   {m} ...', flush=True)
        res = metric_main.calc_metric(
            metric=m,
            G=G_cf,
            dataset_kwargs=dnnlib.EasyDict(
                class_name='training.dataset.ImageFolderDataset',
                path=DATA,
                use_labels=False,
                xflip=True,
                resolution=128,
            ),
            num_gpus=1, rank=0, device=device, cache=True,
        )
        print(f'   {m} = {res.results}   ({res.total_time_str})')
        rec = dict(
            results=dict(res.results), metric=m,
            total_time=res.total_time, total_time_str=res.total_time_str,
            num_gpus=1, snapshot_pkl=f'cf_pokemon128_{vname}',
            baseline_dir=BASELINE_DIR, variant_dir=vdir, timestamp=0,
        )
        with open(out_jsonl, 'w') as f:
            f.write(json.dumps(rec) + '\n')
        print(f'   wrote {out_jsonl}')
    del G_cf
    torch.cuda.empty_cache()

print('\nAll done.')
