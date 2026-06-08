#!/usr/bin/env python
"""Pokemon128 drift summary: per-layer + run-level + comparison vs ImageNet."""
import os, json

REPO = '/home/ygryu/log1/stylegan-xl'

POK_METRICS = {
    'baseline': {'fid': 58.516, 'kid': 0.0328, 'p': 0.549, 'r': 0.011},
    'a':        {'fid': 54.836, 'kid': 0.0301, 'p': 0.530, 'r': 0.024},
    'a2':       {'fid': 56.181, 'kid': 0.0309, 'p': 0.573, 'r': 0.008},
    'a3':       {'fid': 54.034, 'kid': 0.0255, 'p': 0.587, 'r': 0.011},
    'b':        {'fid': 51.477, 'kid': 0.0265, 'p': 0.642, 'r': 0.011},
    'c':        {'fid': 51.645, 'kid': 0.0266, 'p': 0.615, 'r': 0.012},
    'd_cosine': {'fid': 53.817, 'kid': 0.0281, 'p': 0.591, 'r': 0.012},
    'd_linear': {'fid': 53.969, 'kid': 0.0291, 'p': 0.545, 'r': 0.031},
}

# ImageNet sr128 reference (already computed)
IMG_SR128 = {
    'baseline': {'fid': 53.078, 'mean_drift': 0.0025},
    'a':        {'fid': 55.096, 'mean_drift': 0.0051},
    'a2':       {'fid': 52.967, 'mean_drift': 0.0074},
    'a3':       {'fid': 55.900, 'mean_drift': 0.0097},
    'c':        {'fid': 54.833, 'mean_drift': 0.0056},
    'd_cosine': {'fid': 56.328, 'mean_drift': 0.0046},
    'd_linear': {'fid': 54.817, 'mean_drift': 0.0044},
}

results = json.load(open(os.path.join(REPO, 'pokemon_drift_results.json')))


def summarize(layer_dict):
    conv = [m['rel_drift'] for k, m in layer_dict.items() if 'affine' not in k]
    aff  = [m['rel_drift'] for k, m in layer_dict.items() if 'affine' in k]
    sr   = [m['stable_rank'] for k, m in layer_dict.items()
            if 'affine' not in k and m['stable_rank'] > 0]
    return {
        'mean_conv': sum(conv) / max(len(conv), 1),
        'max_conv':  max(conv) if conv else 0,
        'mean_affine': sum(aff) / max(len(aff), 1),
        'mean_sr': sum(sr) / max(len(sr), 1),
    }


order = ['baseline', 'a', 'a2', 'a3', 'b', 'c', 'd_cosine', 'd_linear']

# Run-level table
print("=" * 100)
print(f"{'variant':10} {'FID':>6} {'KID':>7} {'prec':>5} "
      f"{'meanΔ_c':>9} {'maxΔ_c':>8} {'meanΔ_a':>9} {'meanSR':>8}")
print("=" * 100)
for v in order:
    m = POK_METRICS[v]
    if v == 'baseline':
        # baseline = stem reference itself → drift = 0
        print(f"{v:10} {m['fid']:6.2f} {m['kid']:7.4f} {m['p']:5.3f} "
              f"{'0.0000':>9} {'0.0000':>8} {'0.0000':>9} {'—':>8}  (= W^stem reference)")
        continue
    s = summarize(results[v])
    print(f"{v:10} {m['fid']:6.2f} {m['kid']:7.4f} {m['p']:5.3f} "
          f"{s['mean_conv']:9.4f} {s['max_conv']:8.4f} "
          f"{s['mean_affine']:9.4f} {s['mean_sr']:8.2f}")

print()
print("Per-layer CONV .weight drift (relative Frobenius, head excluded)")
print("=" * 100)
# Collect layer names
layer_set = set()
for v, layers in results.items():
    for k, m in layers.items():
        if 'affine' not in k and m.get('weight_frob', 0) > 0:
            layer_set.add(m['layer_name'])
def lkey(n):
    try:
        return int(n.split('_')[0][1:])
    except Exception:
        return 99
layers = sorted(layer_set, key=lkey)

variants = [v for v in order if v != 'baseline' and v in results]
header = f"{'layer':14}" + ''.join(f"{v:>10}" for v in variants)
print(header)
for L in layers:
    row = f"{L:14}"
    for v in variants:
        drifts = [m['rel_drift'] for k, m in results[v].items()
                  if m.get('layer_name') == L and 'affine' not in k]
        d = drifts[0] if drifts else 0.0
        row += f"  {d:8.4f}"
    print(row)

# Stable rank of last 3 stem layers
print()
print("=" * 100)
print("Stable rank of conv ΔW for last 3 stem layers (variant column)")
print("=" * 100)
last3 = layers[-3:]
header = f"{'layer':14}" + ''.join(f"{v:>10}" for v in variants)
print(header)
for L in last3:
    row = f"{L:14}"
    for v in variants:
        srs = [m['stable_rank'] for k, m in results[v].items()
               if m.get('layer_name') == L and 'affine' not in k]
        sr = srs[0] if srs else 0.0
        row += f"  {sr:8.2f}"
    print(row)

# Cross-dataset comparison
print()
print("=" * 100)
print("Pokemon128 (underfit stem) vs ImageNet sr128 (converged stem)")
print("Mean conv drift, same variant, same stage resolution (128² output)")
print("=" * 100)
print(f"{'variant':10} {'pokémon Δ':>12} {'imagenet Δ':>13} {'ratio':>10}  "
      f"{'pokémon FID':>13} {'imagenet FID':>14}")
for v in order:
    pok = POK_METRICS.get(v, {})
    img = IMG_SR128.get(v, {})
    if v == 'baseline':
        pokΔ = 0.0
    elif v in results:
        s = summarize(results[v])
        pokΔ = s['mean_conv']
    else:
        pokΔ = None
    imgΔ = img.get('mean_drift')
    ratio = (pokΔ / imgΔ) if (pokΔ is not None and imgΔ and imgΔ > 0) else None
    pok_fid_str = f"{pok.get('fid'):13.2f}" if pok.get('fid') else f"{'—':>13}"
    img_fid_str = f"{img.get('fid'):14.2f}" if img.get('fid') else f"{'—':>14}"
    pokΔ_str = f"{pokΔ:12.4f}" if pokΔ is not None else f"{'—':>12}"
    imgΔ_str = f"{imgΔ:13.4f}" if imgΔ is not None else f"{'—':>13}"
    ratio_str = f"{ratio:10.2f}x" if ratio is not None else f"{'—':>11}"
    print(f"{v:10} {pokΔ_str} {imgΔ_str} {ratio_str}  {pok_fid_str} {img_fid_str}")
