#!/usr/bin/env python
"""Side-by-side: same variant, same stage (64->128), different stem state.

Pokemon128 runs = UNDERFIT stem (pokemon64_test100, from-scratch low-compute).
ImageNet sr128 runs = CONVERGED stem (pretrained/imagenet64.pkl, paper 87M kimg).

For each shared variant prints:
  - per-layer conv drift in both, aligned by 'distance from last stem layer'
  - run-level summary
  - ratio (underfit / converged)
"""
import os, json

REPO = '/home/ygryu/log1/stylegan-xl'

POK = json.load(open(os.path.join(REPO, 'pokemon_drift_results.json')))  # {variant: {layer_param: metrics}}
IMG = json.load(open(os.path.join(REPO, 'analysis_results.json')))       # {stage: {variant: {layer_param: metrics}}}

# pokemon variant key -> imagenet sr128 variant key
SHARED = [
    ('a',        'a'),
    ('a2',       'a2'),
    ('a3',       'a3'),
    ('c',        'c'),
    ('d_cosine', 'd_cosine'),
    ('d_linear', 'd_linear'),
]


def layer_order(layer_dict):
    """Return ordered list of stem layer names by their L<idx>_ prefix."""
    seen = []
    for k, m in layer_dict.items():
        if 'affine' in k:
            continue
        L = m.get('layer_name')
        if L and L not in seen:
            seen.append(L)
    def lkey(n):
        try:
            return int(n.split('_')[0][1:])
        except Exception:
            return 99
    return sorted(seen, key=lkey)


def conv_drift(layer_dict, layer_name):
    for k, m in layer_dict.items():
        if 'affine' in k:
            continue
        if m.get('layer_name') == layer_name:
            return m['rel_drift']
    return None


def fmt(x):
    return f"{x:7.4f}" if isinstance(x, float) else f"{x:>7}"


for pok_v, img_v in SHARED:
    print('=' * 90)
    print(f"Variant {pok_v.upper():<12} — same variant, both 64²→128² super-res, head_layers differ")
    print(f"  underfit stem (Pokemon128)   vs   converged stem (ImageNet sr128)")
    print('=' * 90)
    pok_layers = layer_order(POK[pok_v])
    img_layers = layer_order(IMG['sr128'][img_v])
    print(f"  pokemon stem has {len(pok_layers)} stem conv layers; imagenet sr128 stem has {len(img_layers)}")
    print(f"  (head excluded both sides)")
    print()
    print(f"  {'pos':>5} | {'pokemon layer':14} {'pokemon Δ':>10}  | {'imagenet layer':14} {'imagenet Δ':>10}  | {'ratio (pok/img)':>15}")
    print('  ' + '-' * 86)
    # Align from the LAST stem layer backwards
    max_depth = max(len(pok_layers), len(img_layers))
    for d in range(max_depth):
        # d=0 = last stem layer
        pos = f"-{d+1}"
        pL = pok_layers[-(d+1)] if d < len(pok_layers) else None
        iL = img_layers[-(d+1)] if d < len(img_layers) else None
        pD = conv_drift(POK[pok_v], pL) if pL else None
        iD = conv_drift(IMG['sr128'][img_v], iL) if iL else None
        pD_s = f"{pD:10.4f}" if pD is not None else f"{'—':>10}"
        iD_s = f"{iD:10.4f}" if iD is not None else f"{'—':>10}"
        if pD is not None and iD is not None and iD > 1e-8:
            ratio = pD / iD
            r_s = f"{ratio:13.2f}x"
        elif pD is not None and iD is not None and iD <= 1e-8 and pD <= 1e-8:
            r_s = f"{'(both 0)':>14}"
        elif pD is not None and pD > 1e-8 and (iD is None or iD <= 1e-8):
            r_s = f"{'(img≈0)':>14}"
        elif iD is not None and iD > 1e-8 and (pD is None or pD <= 1e-8):
            r_s = f"{'(pok≈0)':>14}"
        else:
            r_s = f"{'—':>14}"
        pL_s = pL or '(no layer)'
        iL_s = iL or '(no layer)'
        print(f"  {pos:>5} | {pL_s:14} {pD_s}  | {iL_s:14} {iD_s}  | {r_s:>15}")
    print()
    # summary
    pok_conv = [v['rel_drift'] for k, v in POK[pok_v].items() if 'affine' not in k]
    img_conv = [v['rel_drift'] for k, v in IMG['sr128'][img_v].items() if 'affine' not in k]
    pm = sum(pok_conv) / max(len(pok_conv), 1)
    im = sum(img_conv) / max(len(img_conv), 1)
    print(f"  mean conv drift across all stem layers:")
    print(f"     pokemon  (underfit):  {pm:.4f}")
    print(f"     imagenet (converged): {im:.4f}")
    print(f"     ratio:                {pm/im:.2f}x" if im > 1e-8 else "")
    print()
