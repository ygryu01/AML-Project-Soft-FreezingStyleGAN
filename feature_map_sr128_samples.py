#!/usr/bin/env python
"""sr128 feature-map analysis on 10 different (z, class) samples.

For each sample saves one PNG in the same format as feature_maps_sr128.png:
   rows = baseline + 6 variants (a, a2, a3, c, d_cosine, d_linear)
   cols = synthesis layer (channel-mean feature map) + final generated image
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

import numpy as np, torch
import matplotlib.pyplot as plt
import dnnlib, legacy

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
SUBDIR = '00000-stylegan3-t-imagenet100_128-gpus1-batch32'

VARIANTS = [
    ('baseline', 'imagenet100_128_conv_baseline'),
    ('a',        'imagenet100_128_conv_a'),
    ('a2',       'imagenet100_128_conv_a2'),
    ('a3',       'imagenet100_128_conv_a3'),
    ('c',        'imagenet100_128_conv_c'),
    ('d_cosine', 'imagenet100_128_variant_d_cosine'),
    ('d_linear', 'imagenet100_128_variant_d_linear'),
]

N_SAMPLES = 10
# Class fixed per run. SAMPLE_CLASS env var overrides default.
CLASS_IDX = int(os.environ.get('SAMPLE_CLASS', '7'))
# Output filename suffix. Defaults to empty (= original chicken naming).
SUFFIX = os.environ.get('SAMPLE_SUFFIX', '')
SAMPLES = [(s, CLASS_IDX) for s in range(N_SAMPLES)]


def load_G(path):
    with open(path, 'rb') as f:
        return legacy.load_network_pkl(f)['G_ema'].eval().to(device).requires_grad_(False)


def img_to_display(arr):
    a = (arr[0].transpose(1, 2, 0) + 1.0) / 2.0
    return np.clip(a, 0, 1)


def capture(G, z, c):
    feats = {}
    handles = []
    for name in G.synthesis.layer_names:
        def hook(m, inp, out, n=name):
            feats[n] = out.detach().float().cpu().numpy()
        handles.append(getattr(G.synthesis, name).register_forward_hook(hook))
    with torch.no_grad():
        img = G(z, c, noise_mode='const')
    for h in handles:
        h.remove()
    return img.detach().cpu().numpy(), feats


# Pre-load all 7 G_ema (each ~600 MB CPU). Total ~4 GB.
print('Pre-loading 7 variant G_ema (CPU)...')
Gs = {}
for v, d in VARIANTS:
    p = os.path.join(RUNDIR, d, SUBDIR, 'network-snapshot.pkl')
    print(f'  {v}')
    Gs[v] = load_G(p)
G0 = Gs[VARIANTS[0][0]]
z_dim = G0.z_dim; c_dim = G0.c_dim
layer_names = list(G0.synthesis.layer_names)
head_set = set(getattr(G0, 'head_layer_names', []) or [])
print(f'  layers={len(layer_names)}  z_dim={z_dim}  c_dim={c_dim}')

# Process each sample
for sample_idx, (seed, class_idx) in enumerate(SAMPLES):
    torch.manual_seed(seed)
    z = torch.randn(1, z_dim, device=device)
    c = torch.zeros(1, c_dim, device=device)
    c[0, class_idx] = 1.0
    print(f'\n=== sample {sample_idx:02d}  seed={seed}  class={class_idx} ===')

    feats_all = {}
    imgs_all  = {}
    for v, _ in VARIANTS:
        print(f'  forward {v}...')
        img_v, fv = capture(Gs[v], z, c)
        feats_all[v] = fv
        imgs_all[v]  = img_v

    # Plot
    n_v = len(VARIANTS)
    n_l = len(layer_names)
    fig, axes = plt.subplots(n_v, n_l + 1, figsize=(1.45 * (n_l + 1), 1.7 * n_v))

    for ci, name in enumerate(layer_names):
        panels = []
        for v, _ in VARIANTS:
            f = feats_all[v][name][0]
            panels.append(f.mean(axis=0) if f.ndim == 3 else None)
        non_none = [p for p in panels if p is not None]
        if not non_none:
            continue
        vmin = min(p.min() for p in non_none)
        vmax = max(p.max() for p in non_none)
        for ri, (v, _) in enumerate(VARIANTS):
            ax = axes[ri, ci]
            p = panels[ri]
            if p is None:
                ax.set_visible(False)
            else:
                ax.imshow(p, vmin=vmin, vmax=vmax, cmap='viridis')
            ax.set_xticks([]); ax.set_yticks([])
        shape0 = feats_all[VARIANTS[0][0]][name][0].shape
        role = 'head' if name in head_set else 'stem'
        axes[0, ci].set_title(f'{name}\n({role}, {shape0[1]}×{shape0[2]})', fontsize=6)

    for ri, (v, _) in enumerate(VARIANTS):
        ax = axes[ri, -1]
        ax.imshow(img_to_display(imgs_all[v]))
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, -1].set_title('generated', fontsize=7)

    for ri, (v, _) in enumerate(VARIANTS):
        axes[ri, 0].set_ylabel(v, fontsize=10)

    fig.suptitle(f'sr128 — sample {sample_idx:02d}  (z seed={seed}, class={class_idx})',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(REPO, f'feature_maps_sr128{SUFFIX}_s{sample_idx:02d}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')

print('\nAll 10 sample PNGs done.')
