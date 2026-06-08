#!/usr/bin/env python
"""Layer-wise drift and ΔW spectrum analysis for the freeze-policy ablation.

For each (stage, variant) run, compares the trained G_ema.synthesis weights
against the pretrained stem pkl (the W^stem that the run started from), and
computes per-layer drift + ΔW spectrum metrics.

Outputs JSON to analysis_results.json and a markdown summary to
analysis_summary.md.
"""
import sys, os, json, math
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

def _find_snap(run_dir):
    if not os.path.isdir(run_dir):
        return None
    for sub in sorted(os.listdir(run_dir)):
        cand = os.path.join(run_dir, sub, 'network-snapshot.pkl')
        if os.path.isfile(cand):
            return cand
    return None


# (stage, variant) -> trained snapshot path
RUNS = []
for stage, outdim in [('sr32', 32), ('sr64', 64), ('sr128', 128)]:
    for v in ['baseline', 'a', 'a2', 'a3', 'b', 'c']:
        snap = _find_snap(os.path.join(RUNDIR, f'imagenet100_{outdim}_conv_{v}'))
        if snap is not None:
            RUNS.append((stage, v, snap))

# Variant D (soft-freeze + LR annealing) — only sr128 in this machine.
for v in ['d_cosine', 'd_linear']:
    snap = _find_snap(os.path.join(RUNDIR, f'imagenet100_128_variant_{v}'))
    if snap is not None:
        RUNS.append(('sr128', v, snap))

# sr256 runs exist but stem reference (imagenet128.pkl) is NOT present on
# this machine, so they're skipped — see report block at the end.
SR256_SKIPPED = []
for v in ['baseline', 'variant_a', 'variant_a2', 'variant_a3']:
    snap = _find_snap(os.path.join(RUNDIR, f'imagenet100_256_converged_{v}'))
    if snap is not None:
        SR256_SKIPPED.append(('sr256', v, snap))


def load_G(pkl_path, key='G_ema'):
    with open(pkl_path, 'rb') as f:
        data = legacy.load_network_pkl(f)
    return data[key]


def spectrum_metrics(dW):
    """Compute spectrum stats for a weight delta tensor.

    Reshapes conv weights to Cout x (Cin*k*k); 2D weights pass through.
    Returns dict with frob, rel-only-set-by-caller, top1_energy, stable_rank,
    entropy_rank, spec_norm, n_singular.
    """
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


def analyze_run(G_stem, G_trained):
    """Compare two G.synthesis state dicts on STEM-only layers.

    Excludes any layer name listed in G_trained.head_layer_names (these were
    re-initialized as super-res head and are not stem-policy targets).
    Returns dict {layer_param: metrics}.
    """
    sd0 = G_stem.synthesis.state_dict()
    sd1 = G_trained.synthesis.state_dict()
    head_names = set(getattr(G_trained, 'head_layer_names', []) or [])
    out = {}
    for k in sorted(set(sd0) & set(sd1)):
        # k looks like "L14_52_1024.weight" — first component is the layer name
        layer_name = k.split('.', 1)[0]
        if layer_name in head_names:
            continue
        W0, W1 = sd0[k], sd1[k]
        if W0.shape != W1.shape:
            continue
        if W0.dim() < 2:
            continue
        dW = (W1 - W0).detach().to(torch.float32)
        W0f = W0.detach().to(torch.float32)
        ref = float(W0f.norm().item())
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
    cache = {}
    results = {}
    for stage, variant, snap in RUNS:
        print(f'[{stage} {variant}] loading...', flush=True)
        if stage not in cache:
            cache[stage] = load_G(STEMS[stage])
        G_stem = cache[stage]
        G_tr = load_G(snap)
        per_layer = analyze_run(G_stem, G_tr)
        results.setdefault(stage, {})[variant] = per_layer
        # quick summary line
        drifts = [m['rel_drift'] for m in per_layer.values() if 'weight' in m.get('shape', []) or True]
        mean_d = sum(drifts) / max(len(drifts), 1)
        max_d = max(drifts) if drifts else 0
        print(f'   layers={len(per_layer)}  mean_drift={mean_d:.4f}  max_drift={max_d:.4f}', flush=True)

    out_json = os.path.join(REPO, 'analysis_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nWrote {out_json}')
    if SR256_SKIPPED:
        print(f'\n[skipped] {len(SR256_SKIPPED)} sr256 run(s) — stem reference '
              f'pretrained/imagenet128.pkl missing on this machine.')
        for s in SR256_SKIPPED:
            print(f'    {s[1]:18}  {s[2]}')


if __name__ == '__main__':
    main()
