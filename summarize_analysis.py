#!/usr/bin/env python
"""Summarize drift+spectrum analysis with metric results.

Produces:
  - Run-level summary table (FID, drift mean/max, spectrum aggregates)
  - Per-stage per-layer drift heatmap (text)
  - B vs C profile comparison
"""
import os, json, glob

REPO = '/home/ygryu/log1/stylegan-xl'
RUNDIR = os.path.join(REPO, 'training-runs')

# Hard-coded metric results we've collected
METRICS = {
    ('sr32', 'a3'):       {'fid': 58.635, 'kid': 0.0464, 'p': 0.766, 'r': 0.013},
    ('sr32', 'b'):        {'fid': 58.323, 'kid': 0.0460, 'p': 0.783, 'r': 0.014},
    ('sr32', 'c'):        {'fid': 59.098, 'kid': 0.0464, 'p': 0.768, 'r': 0.015},
    ('sr64', 'baseline'): {'fid': 52.956, 'kid': 0.0342, 'p': 0.510, 'r': 0.088},
    ('sr64', 'a'):        {'fid': 46.822, 'kid': 0.0292, 'p': 0.555, 'r': 0.098},
    ('sr64', 'a2'):       {'fid': 48.928, 'kid': 0.0319, 'p': 0.566, 'r': 0.086},
    ('sr64', 'a3'):       {'fid': 49.633, 'kid': 0.0321, 'p': 0.589, 'r': 0.074},
    ('sr64', 'b'):        {'fid': 46.260, 'kid': 0.0290, 'p': 0.545, 'r': 0.095},
    ('sr64', 'c'):        {'fid': 45.327, 'kid': 0.0282, 'p': 0.583, 'r': 0.102},
    ('sr128', 'baseline'):{'fid': 53.078, 'kid': 0.0268, 'p': 0.484, 'r': 0.145},
    ('sr128', 'a'):       {'fid': 55.096, 'kid': 0.0285, 'p': 0.469, 'r': 0.142},
    ('sr128', 'a2'):      {'fid': 52.967, 'kid': 0.0267, 'p': 0.490, 'r': 0.134},
    ('sr128', 'a3'):      {'fid': 55.900, 'kid': 0.0292, 'p': 0.466, 'r': 0.130},
    ('sr128', 'c'):       {'fid': 54.833, 'kid': 0.0284, 'p': 0.478, 'r': 0.136},
    ('sr128', 'd_cosine'):{'fid': 56.328, 'kid': 0.0295, 'p': 0.456, 'r': 0.128},
    ('sr128', 'd_linear'):{'fid': 54.817, 'kid': 0.0278, 'p': 0.465, 'r': 0.136},
}

results = json.load(open(os.path.join(REPO, 'analysis_results.json')))


def per_layer_conv_drift(layer_metrics):
    """Aggregate per-layer drift, separating conv .weight from affine.weight."""
    out = {}  # {layer_name: {'conv': drift, 'affine': drift, 'specnorm_conv': ..}}
    for key, m in layer_metrics.items():
        layer = m['layer_name']
        kind = 'affine' if 'affine' in key else 'conv'
        out.setdefault(layer, {})[kind] = m['rel_drift']
        if kind == 'conv':
            out[layer]['stable_rank'] = m['stable_rank']
            out[layer]['top1_energy'] = m['top1_energy']
            out[layer]['spec_norm'] = m['spec_norm']
    return out


def summarize_run(layer_metrics):
    conv_drifts = [m['rel_drift'] for k, m in layer_metrics.items()
                   if 'affine' not in k]
    affine_drifts = [m['rel_drift'] for k, m in layer_metrics.items()
                     if 'affine' in k]
    stable_ranks = [m['stable_rank'] for k, m in layer_metrics.items()
                    if 'affine' not in k and m['stable_rank'] > 0]
    return {
        'mean_conv_drift': sum(conv_drifts) / max(len(conv_drifts), 1),
        'max_conv_drift':  max(conv_drifts) if conv_drifts else 0,
        'mean_affine_drift': sum(affine_drifts) / max(len(affine_drifts), 1),
        'max_affine_drift':  max(affine_drifts) if affine_drifts else 0,
        'mean_stable_rank': sum(stable_ranks) / max(len(stable_ranks), 1),
    }


def fmt_metrics(m):
    return f"FID={m['fid']:5.2f} KID={m['kid']:.4f} p={m['p']:.3f}"


print("=" * 110)
print(f"{'stage':6} {'variant':10} {'FID':>6} {'KID':>7} {'prec':>5} "
      f"{'meanΔ_c':>9} {'maxΔ_c':>8} {'meanΔ_a':>9} {'meanSR':>8}")
print("=" * 110)
order = ['baseline', 'a', 'a2', 'a3', 'b', 'c', 'd_cosine', 'd_linear']
for stage in ['sr32', 'sr64', 'sr128']:
    if stage not in results:
        continue
    for v in order:
        if v not in results[stage]:
            continue
        m = METRICS.get((stage, v))
        s = summarize_run(results[stage][v])
        if m is None:
            mline = f"{'':5} {'':6} {'':4}"
        else:
            mline = f"{m['fid']:6.2f} {m['kid']:7.4f} {m['p']:5.3f}"
        print(f"{stage:6} {v:10} {mline} "
              f"{s['mean_conv_drift']:9.4f} {s['max_conv_drift']:8.4f} "
              f"{s['mean_affine_drift']:9.4f} {s['mean_stable_rank']:8.2f}")
    print('-' * 110)

print()
print("Per-layer CONV .weight drift (relative Frobenius, head excluded)")
print("=" * 110)

# Per stage, build a layer-by-variant grid
for stage in ['sr32', 'sr64', 'sr128']:
    if stage not in results:
        continue
    # Collect all layer names that appear (sorted by index in name e.g. L0_, L1_, ...)
    layer_set = set()
    for v in results[stage].values():
        for k, m in v.items():
            if 'affine' not in k and m.get('weight_frob', 0) > 0:
                layer_set.add(m['layer_name'])
    def lkey(n):
        try:
            return int(n.split('_')[0][1:])
        except Exception:
            return 99
    layers = sorted(layer_set, key=lkey)
    variants = [v for v in order if v in results[stage]]
    if not variants:
        continue
    print(f"\n### {stage}")
    header = f"{'layer':14}" + ''.join(f"{v:>9}" for v in variants)
    print(header)
    for L in layers:
        row = f"{L:14}"
        for v in variants:
            drifts = [m['rel_drift'] for k, m in results[stage][v].items()
                      if m.get('layer_name') == L and 'affine' not in k]
            d = drifts[0] if drifts else 0.0
            row += f"  {d:7.4f}"
        print(row)

# Spectrum: conv stable-rank for a few key layers (last 3 stem layers)
print()
print("=" * 110)
print("Stable rank of conv ΔW for last 3 stem layers")
print("(high = update spread across many singular directions; low = update is concentrated)")
print("=" * 110)
for stage in ['sr32', 'sr64', 'sr128']:
    if stage not in results:
        continue
    layer_set = set()
    for v in results[stage].values():
        for k, m in v.items():
            if 'affine' not in k:
                layer_set.add(m['layer_name'])
    def lkey(n):
        try:
            return int(n.split('_')[0][1:])
        except Exception:
            return 99
    layers = sorted(layer_set, key=lkey)
    last3 = layers[-3:]
    print(f"\n### {stage}  last 3 stem layers: {last3}")
    variants = [v for v in order if v in results[stage]]
    header = f"{'layer':14}" + ''.join(f"{v:>9}" for v in variants)
    print(header)
    for L in last3:
        row = f"{L:14}"
        for v in variants:
            srs = [m['stable_rank'] for k, m in results[stage][v].items()
                   if m.get('layer_name') == L and 'affine' not in k]
            sr = srs[0] if srs else 0.0
            row += f"  {sr:7.2f}"
        print(row)


# Stage-level aggregate: mean drift across variant FAMILIES per stage
print()
print("=" * 110)
print("Stage-level mean conv drift, grouped by variant family")
print("(reference = paper converged stem; baseline=head-only, A=partial unfreeze,")
print(" B/C=soft-freeze, D=soft-freeze + LR annealing)")
print("=" * 110)
FAMILY = {
    'baseline': 'baseline',
    'a': 'A', 'a2': 'A', 'a3': 'A',
    'b': 'B', 'c': 'C',
    'd_cosine': 'D', 'd_linear': 'D',
}
families = ['baseline', 'A', 'B', 'C', 'D']
print(f"{'stage':6} " + ''.join(f"{f:>10}" for f in families))
for stage in ['sr32', 'sr64', 'sr128']:
    if stage not in results:
        continue
    row = f"{stage:6} "
    for f in families:
        ds = []
        for v, layers in results[stage].items():
            if FAMILY.get(v) != f:
                continue
            cd = [m['rel_drift'] for k, m in layers.items() if 'affine' not in k]
            if cd:
                ds.append(sum(cd) / len(cd))
        if ds:
            mean_ds = sum(ds) / len(ds)
            row += f"{mean_ds:10.4f}"
        else:
            row += f"{'—':>10}"
    print(row)

print()
print("Implied: where stem is more underfit (lower stage), variant families show")
print("a bigger family-vs-baseline drift gap; on sr128 the gap shrinks because the")
print("stem is closer to converged for its target resolution.")
