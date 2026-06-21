#!/usr/bin/env python
"""Unified drift + spectrum analysis for StyleGAN experiments.

Supports three modes:
  general   - sr32/sr64/sr128 stem drift + spectrum checks
  pokemon   - pokemon128 stem drift analysis using head-only baseline
  sr256     - sr256 drift analysis merged into analysis_results.json
  all       - run all available analyses in sequence
"""
import argparse
import importlib.abc
import importlib.machinery
import json
import math
import os
import shutil
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import dnnlib
import legacy

REPO = '/home/ygryu/log1/stylegan-xl'
RUNDIR = os.path.join(REPO, 'training-runs')
STEMS = {
    'sr32':  os.path.join(REPO, 'pretrained/imagenet16.pkl'),
    'sr64':  os.path.join(REPO, 'pretrained/imagenet32.pkl'),
    'sr128': os.path.join(REPO, 'pretrained/imagenet64.pkl'),
}
POKEMON_STEM_REF = os.path.join(
    RUNDIR,
    'pokemon128_test100/00000-stylegan3-t-pokemon128-gpus1-batch32/network-snapshot.pkl',
)
SR256_STEM_REF = os.path.join(REPO, 'pretrained', 'imagenet128.pkl')

GENERAL_VARIANTS = ['baseline', 'a', 'a2', 'a3', 'b', 'c']
GENERAL_SR128_EXTRA = ['d_cosine', 'd_linear']

POKEMON_VARIANTS = [
    ('a',        'pokemon128_variant_a'),
    ('a2',       'pokemon128_variant_a2'),
    ('a3',       'pokemon128_variant_a3'),
    ('b',        'pokemon128_variant_b'),
    ('c',        'pokemon128_variant_c'),
    ('d_cosine', 'pokemon128_variant_d_cosine'),
    ('d_linear', 'pokemon128_variant_d_linear'),
]

SR256_RUN_MAP = [
    ('baseline',  'imagenet100_256_converged_baseline'),
    ('a',         'imagenet100_256_converged_variant_a'),
    ('a2',        'imagenet100_256_converged_variant_a2'),
    ('a3',        'imagenet100_256_converged_variant_a3'),
]


def _find_snap(run_dir):
    if not os.path.isdir(run_dir):
        return None
    for sub in sorted(os.listdir(run_dir)):
        cand = os.path.join(run_dir, sub, 'network-snapshot.pkl')
        if os.path.isfile(cand):
            return cand
    return None


def load_G(pkl_path, key='G_ema'):
    with open(pkl_path, 'rb') as f:
        return legacy.load_network_pkl(f)[key]


def spectrum_metrics(dW):
    if dW.dim() < 2:
        return None
    mat = dW.detach().to(torch.float32).reshape(dW.shape[0], -1)
    if mat.shape[1] == 0:
        return None
    try:
        S = torch.linalg.svdvals(mat)
    except Exception:
        return None
    s2 = (S ** 2)
    total = float(s2.sum().item())
    spec_norm = float(S[0].item())
    if total <= 0 or spec_norm <= 0:
        return dict(
            frob=0.0,
            top1_energy=0.0,
            stable_rank=0.0,
            entropy_rank=0.0,
            spec_norm=0.0,
            n_singular=int(S.numel()),
        )
    p = (s2 / total).clamp_min(1e-30)
    H = float(-(p * p.log()).sum().item())
    return dict(
        frob=float(math.sqrt(total)),
        top1_energy=float(s2[0].item() / total),
        stable_rank=float(total / (spec_norm ** 2)),
        entropy_rank=float(math.exp(H)),
        spec_norm=spec_norm,
        n_singular=int(S.numel()),
    )


def analyze_run(G_stem, G_trained):
    sd0 = G_stem.synthesis.state_dict()
    sd1 = G_trained.synthesis.state_dict()
    head_names = set(getattr(G_trained, 'head_layer_names', []) or [])
    out = {}
    for k in sorted(set(sd0) & set(sd1)):
        layer_name = k.split('.', 1)[0]
        if layer_name in head_names:
            continue
        W0, W1 = sd0[k], sd1[k]
        if W0.shape != W1.shape or W0.dim() < 2:
            continue
        dW = (W1 - W0).detach().to(torch.float32)
        ref = float(W0.detach().to(torch.float32).norm().item())
        frob = float(dW.norm().item())
        rel_drift = frob / (ref + 1e-12)
        spec = spectrum_metrics(dW)
        if spec is None:
            continue
        spec.update(
            rel_drift=rel_drift,
            ref_norm=ref,
            weight_frob=frob,
            shape=list(W0.shape),
            layer_name=layer_name,
        )
        out[k] = spec
    return out


def run_general():
    cache = {}
    results = {}
    sr256_skipped = []

    runs = []
    for stage, outdim in [('sr32', 32), ('sr64', 64), ('sr128', 128)]:
        for v in GENERAL_VARIANTS:
            snap = _find_snap(os.path.join(RUNDIR, f'imagenet100_{outdim}_conv_{v}'))
            if snap is not None:
                runs.append((stage, v, snap))
    for v in GENERAL_SR128_EXTRA:
        snap = _find_snap(os.path.join(RUNDIR, f'imagenet100_128_variant_{v}'))
        if snap is not None:
            runs.append(('sr128', v, snap))

    for stage, variant, snap in runs:
        print(f'[{stage} {variant}] loading...', flush=True)
        if stage not in cache:
            cache[stage] = load_G(STEMS[stage])
        G_stem = cache[stage]
        G_tr = load_G(snap)
        per_layer = analyze_run(G_stem, G_tr)
        results.setdefault(stage, {})[variant] = per_layer
        drifts = [m['rel_drift'] for m in per_layer.values()]
        mean_d = sum(drifts) / max(len(drifts), 1)
        max_d = max(drifts) if drifts else 0
        print(f'   layers={len(per_layer)}  mean_drift={mean_d:.4f}  max_drift={max_d:.4f}', flush=True)

    out_json = os.path.join(REPO, 'analysis_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nWrote {out_json}')

    for v in ['baseline', 'variant_a', 'variant_a2', 'variant_a3']:
        snap = _find_snap(os.path.join(RUNDIR, f'imagenet100_256_converged_{v}'))
        if snap is not None:
            sr256_skipped.append((v, snap))
    if sr256_skipped:
        print(f'\n[skipped] {len(sr256_skipped)} sr256 run(s) — stem reference '
              f'pretrained/imagenet128.pkl missing on this machine.')
        for s in sr256_skipped:
            print(f'    {s[0]:18}  {s[1]}')


def run_pokemon():
    if not os.path.isfile(POKEMON_STEM_REF):
        raise FileNotFoundError(f'Pokemon stem reference missing: {POKEMON_STEM_REF}')

    print('[stem ref] loading pokemon128_test100 (head-only baseline)...')
    G_stem = load_G(POKEMON_STEM_REF)
    print(f'           head_layer_names = {getattr(G_stem, "head_layer_names", None)}')

    results = {}
    for vname, dirname in POKEMON_VARIANTS:
        snap = os.path.join(RUNDIR, dirname,
                            '00000-stylegan3-t-pokemon128-gpus1-batch32',
                            'network-snapshot.pkl')
        if not os.path.isfile(snap):
            print(f'[skip] {vname} — snapshot missing')
            continue
        print(f'[{vname}] loading...', flush=True)
        G_tr = load_G(snap)
        per_layer = analyze_run(G_stem, G_tr)
        results[vname] = per_layer
        conv_d = [m['rel_drift'] for k, m in per_layer.items() if 'affine' not in k]
        affine_d = [m['rel_drift'] for k, m in per_layer.items() if 'affine' in k]
        mean_c = sum(conv_d) / max(len(conv_d), 1)
        max_c = max(conv_d) if conv_d else 0
        mean_a = sum(affine_d) / max(len(affine_d), 1)
        print(f'   conv layers={len(conv_d)} ' \
              f'mean_drift={mean_c:.4f} max={max_c:.4f}  ' \
              f'affine mean={mean_a:.4f}', flush=True)

    out_json = os.path.join(REPO, 'pokemon_drift_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nWrote {out_json}')


class _DummyBase:
    def __new__(cls, *a, **k):
        return object.__new__(cls)
    def __init__(self, *a, **k):
        pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


def _dummy_class(name):
    return type(name, (_DummyBase,), {})


class _DummyTimmFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith('timm.'):
            return None
        return importlib.machinery.ModuleSpec(fullname, self)
    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__getattr__ = _dummy_class
        m.__path__ = []
        return m
    def exec_module(self, module):
        pass


def install_sr256_fallback():
    if not any(isinstance(f, _DummyTimmFinder) for f in sys.meta_path):
        sys.meta_path.append(_DummyTimmFinder())


def run_sr256():
    assert os.path.isfile(SR256_STEM_REF), f'missing stem: {SR256_STEM_REF}'
    install_sr256_fallback()

    out_json = os.path.join(REPO, 'analysis_results.json')
    results = {} if not os.path.isfile(out_json) else json.load(open(out_json))

    G_stem = load_G(SR256_STEM_REF)
    sr256 = {}
    for var, rd in SR256_RUN_MAP:
        snap = _find_snap(os.path.join(RUNDIR, rd))
        if snap is None:
            print(f'[sr256 {var}] no snapshot -> skip')
            continue
        print(f'[sr256 {var}] loading {snap}', flush=True)
        try:
            G_tr = load_G(snap)
        except Exception as e:
            print(f'   FAILED to load ({type(e).__name__}: {e}) -> skip '
                  f'[snapshot is likely truncated/corrupt]', flush=True)
            continue
        per_layer = analyze_run(G_stem, G_tr)
        sr256[var] = per_layer
        cd = [m['rel_drift'] for k, m in per_layer.items() if 'affine' not in k]
        mean_d = sum(cd) / max(len(cd), 1)
        max_d = max(cd) if cd else 0
        print(f'   layers={len(per_layer)}  mean_conv_drift={mean_d:.4f}  max={max_d:.4f}', flush=True)

    if not sr256:
        print('No sr256 runs analyzed; nothing written.')
        return

    if not os.path.isfile(out_json):
        print(f'Warning: {out_json} missing, creating new analysis_results.json')
    if 'sr256' in results:
        print('Overwriting existing sr256 block in analysis_results.json')
    results['sr256'] = sr256
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Merged sr256 ({list(sr256)}) into {out_json}')


def main():
    parser = argparse.ArgumentParser(description='Unified drift + spectrum analysis')
    parser.add_argument('mode', nargs='?', default='all',
                        choices=['all', 'general', 'pokemon', 'sr256'])
    args = parser.parse_args()

    if args.mode == 'all':
        run_general()
        try:
            run_pokemon()
        except FileNotFoundError as e:
            print(f'[pokemon] skipped: {e}')
        try:
            run_sr256()
        except AssertionError as e:
            print(f'[sr256] skipped: {e}')
    elif args.mode == 'general':
        run_general()
    elif args.mode == 'pokemon':
        run_pokemon()
    elif args.mode == 'sr256':
        run_sr256()


if __name__ == '__main__':
    main()
