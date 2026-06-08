#!/usr/bin/env python
"""Counterfactual feature injection for sr128 — variant head + baseline stem.

For each sample (seed, class):
  - baseline forward, capture last-stem-layer output (L18_84_1024)
  - for each variant:
       (a) normal forward      H_v(S_v(z,c))
       (b) counterfactual fwd  H_v(S_0(z,c))   — inject baseline's L18 output

Outputs:
  counterfactual_sr128{SUFFIX}_s{NN}.png
  rows = baseline + 6 variants;  cols = [normal, counterfactual]
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
BOUNDARY_LAYER = 'L18_84_1024'   # last stem layer in sr128

N_SAMPLES = 10
CLASS_IDX = int(os.environ.get('SAMPLE_CLASS', '7'))
SUFFIX = os.environ.get('SAMPLE_SUFFIX', '')
SAMPLES = [(s, CLASS_IDX) for s in range(N_SAMPLES)]


def load_G(path):
    with open(path, 'rb') as f:
        return legacy.load_network_pkl(f)['G_ema'].eval().to(device).requires_grad_(False)


def img_to_display(arr):
    a = (arr[0].transpose(1, 2, 0) + 1.0) / 2.0
    return np.clip(a, 0, 1)


def forward_capture(G, z, c, layer_name):
    """Run G(z,c) and return (img_np, layer_output_tensor)."""
    captured = {}
    layer = getattr(G.synthesis, layer_name)
    def hook(m, inp, out):
        captured['out'] = out.detach().clone()
    h = layer.register_forward_hook(hook)
    with torch.no_grad():
        img = G(z, c, noise_mode='const')
    h.remove()
    return img.detach().cpu().numpy(), captured['out']


def forward_inject(G, z, c, layer_name, replacement):
    """Run G(z,c) but at layer `layer_name`, replace output with `replacement`."""
    layer = getattr(G.synthesis, layer_name)
    def hook(m, inp, out):
        return replacement
    h = layer.register_forward_hook(hook)
    with torch.no_grad():
        img = G(z, c, noise_mode='const')
    h.remove()
    return img.detach().cpu().numpy()


print('Pre-loading 7 variant G_ema (GPU)...')
Gs = {}
for v, d in VARIANTS:
    p = os.path.join(RUNDIR, d, SUBDIR, 'network-snapshot.pkl')
    print(f'  {v}')
    Gs[v] = load_G(p)
G0 = Gs['baseline']
z_dim = G0.z_dim; c_dim = G0.c_dim
print(f'  z_dim={z_dim}  c_dim={c_dim}  boundary={BOUNDARY_LAYER}')


for sample_idx, (seed, class_idx) in enumerate(SAMPLES):
    torch.manual_seed(seed)
    z = torch.randn(1, z_dim, device=device)
    c = torch.zeros(1, c_dim, device=device); c[0, class_idx] = 1.0
    print(f'\n=== sample {sample_idx:02d}  seed={seed}  class={class_idx} ===')

    # 1) baseline forward + capture boundary feature
    img_base, feat0 = forward_capture(G0, z, c, BOUNDARY_LAYER)
    print(f'  baseline ok  (boundary shape={tuple(feat0.shape)})')

    # 2) each variant: normal + counterfactual
    results = {'baseline': {'normal': img_base, 'cf': img_base}}  # cf identical for baseline
    for v, _ in VARIANTS[1:]:
        # normal
        img_n, _ = forward_capture(Gs[v], z, c, BOUNDARY_LAYER)
        # counterfactual: inject baseline boundary feature
        img_cf = forward_inject(Gs[v], z, c, BOUNDARY_LAYER, feat0)
        results[v] = {'normal': img_n, 'cf': img_cf}
        print(f'  {v} ok')

    # Plot
    n_rows = len(VARIANTS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(5.6, 1.9 * n_rows))
    for ri, (v, _) in enumerate(VARIANTS):
        axes[ri, 0].imshow(img_to_display(results[v]['normal']))
        axes[ri, 1].imshow(img_to_display(results[v]['cf']))
        axes[ri, 0].set_xticks([]); axes[ri, 0].set_yticks([])
        axes[ri, 1].set_xticks([]); axes[ri, 1].set_yticks([])
        axes[ri, 0].set_ylabel(v, fontsize=10)
    axes[0, 0].set_title('normal  $H_v(S_v(z))$', fontsize=9)
    axes[0, 1].set_title('counterfactual  $H_v(S_0(z))$', fontsize=9)
    fig.suptitle(f'sr128 counterfactual injection — sample {sample_idx:02d} '
                 f'(seed={seed}, class={class_idx})\n'
                 f'inject boundary = {BOUNDARY_LAYER} output',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(REPO, f'counterfactual_sr128{SUFFIX}_s{sample_idx:02d}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')

print('\nAll done.')
