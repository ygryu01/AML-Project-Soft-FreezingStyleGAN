#!/usr/bin/env python
"""Pokemon128 per-layer drift analysis.

The original from-scratch underfit stem (pokemon64_test100 snapshot) is not on
this machine. However, pokemon128_test100 is the head-only baseline (stem
frozen), so its G_ema.synthesis stem-layer weights equal the original stem
weights. We use it as the W^stem reference.

Targets: 7 variant runs under pokemon128 (a/a2/a3/b/c/d_cosine/d_linear).
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import dnnlib
import legacy

REPO = '/home/ygryu/log1/stylegan-xl'
RUNDIR = os.path.join(REPO, 'training-runs')

STEM_REF = os.path.join(
    RUNDIR,
    'pokemon128_test100/00000-stylegan3-t-pokemon128-gpus1-batch32/network-snapshot.pkl',
)

VARIANTS = [
    ('a',        'pokemon128_variant_a'),
    ('a2',       'pokemon128_variant_a2'),
    ('a3',       'pokemon128_variant_a3'),
    ('b',        'pokemon128_variant_b'),
    ('c',        'pokemon128_variant_c'),
    ('d_cosine', 'pokemon128_variant_d_cosine'),
    ('d_linear', 'pokemon128_variant_d_linear'),
]


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
        return dict(frob=0.0, top1_energy=0.0, stable_rank=0.0,
                    entropy_rank=0.0, spec_norm=0.0, n_singular=int(S.numel()))
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


def analyze(G_stem, G_trained):
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
        spec['rel_drift'] = rel_drift
        spec['ref_norm'] = ref
        spec['weight_frob'] = frob
        spec['shape'] = list(W0.shape)
        spec['layer_name'] = layer_name
        out[k] = spec
    return out


def main():
    print('[stem ref] loading pokemon128_test100 (head-only baseline)...')
    G_stem = load_G(STEM_REF)
    print(f'           head_layer_names = {getattr(G_stem, "head_layer_names", None)}')

    results = {}
    for vname, dirname in VARIANTS:
        snap = os.path.join(RUNDIR, dirname,
                            '00000-stylegan3-t-pokemon128-gpus1-batch32',
                            'network-snapshot.pkl')
        if not os.path.isfile(snap):
            print(f'[skip] {vname} — snapshot missing')
            continue
        print(f'[{vname}] loading...', flush=True)
        G_tr = load_G(snap)
        per_layer = analyze(G_stem, G_tr)
        results[vname] = per_layer
        conv_d = [m['rel_drift'] for k, m in per_layer.items()
                  if 'affine' not in k]
        affine_d = [m['rel_drift'] for k, m in per_layer.items()
                    if 'affine' in k]
        mean_c = sum(conv_d) / max(len(conv_d), 1)
        max_c = max(conv_d) if conv_d else 0
        mean_a = sum(affine_d) / max(len(affine_d), 1)
        print(f'   conv layers={len(conv_d)} '
              f'mean_drift={mean_c:.4f} max={max_c:.4f}  '
              f'affine mean={mean_a:.4f}', flush=True)

    out_json = os.path.join(REPO, 'pokemon_drift_results.json')
    json.dump(results, open(out_json, 'w'), indent=2)
    print(f'\nWrote {out_json}')


if __name__ == '__main__':
    main()
