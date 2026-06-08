#!/usr/bin/env python
"""For each variant, one small table:
   rows = {underfit, converged}, columns = layer position (-1 = last stem layer),
   values = relative Frobenius drift.
"""
import os, json

REPO = '/home/ygryu/log1/stylegan-xl'
POK = json.load(open(os.path.join(REPO, 'pokemon_drift_results.json')))
IMG = json.load(open(os.path.join(REPO, 'analysis_results.json')))['sr128']

VARIANTS = ['a', 'a2', 'a3', 'c', 'd_cosine', 'd_linear']

def stem_layers(layer_dict):
    seen = []
    for k, m in layer_dict.items():
        if 'affine' in k: continue
        L = m.get('layer_name')
        if L and L not in seen: seen.append(L)
    def lkey(n):
        try: return int(n.split('_')[0][1:])
        except: return 99
    return sorted(seen, key=lkey)

def drift_of(layer_dict, name):
    for k, m in layer_dict.items():
        if 'affine' not in k and m.get('layer_name') == name:
            return m['rel_drift']
    return 0.0

# canonical layer ordering for each side (pokemon has 9, imagenet has 14)
POK_LAYERS = stem_layers(POK['a'])
IMG_LAYERS = stem_layers(IMG['baseline'])
n_pok = len(POK_LAYERS)
n_img = len(IMG_LAYERS)
n_cols = max(n_pok, n_img)

print(f"Pokemon stem (underfit):   {n_pok} layers — {POK_LAYERS}")
print(f"ImageNet stem (converged): {n_img} layers — {IMG_LAYERS}")
print()
print("Layer columns are 'distance from last stem layer'. -1 = the last stem layer "
      "(closest to head); -2 = one before, etc.\n"
      "Pokemon shorter; cells beyond its depth marked '—'.")
print()

for v in VARIANTS:
    print('=' * (15 + n_cols * 9))
    print(f'Variant {v.upper()}')
    print('=' * (15 + n_cols * 9))
    # Header
    header = f"{'':12}"
    for d in range(n_cols):
        header += f"{'-'+str(d+1):>9}"
    print(header)
    # Underfit row
    row_u = f"{'underfit':12}"
    for d in range(n_cols):
        if d < n_pok:
            L = POK_LAYERS[-(d+1)]
            val = drift_of(POK[v], L)
            row_u += f"{val:9.4f}"
        else:
            row_u += f"{'—':>9}"
    print(row_u)
    # Converged row
    row_c = f"{'converged':12}"
    for d in range(n_cols):
        if d < n_img:
            L = IMG_LAYERS[-(d+1)]
            val = drift_of(IMG[v], L)
            row_c += f"{val:9.4f}"
        else:
            row_c += f"{'—':>9}"
    print(row_c)
    print()
