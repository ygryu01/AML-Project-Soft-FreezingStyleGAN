#!/usr/bin/env python
"""Feature-map visualization: sr128 baseline vs variant a3, same latent.

For a fixed (z, class) pair, forward-pass both generators and capture every
synthesis layer's output via forward hooks. Then visualize each layer as the
channel-mean feature map (averaged across C_out channels of that layer).

Outputs:
  feature_maps_sr128.png   — N_layers × 2 (baseline row, a3 row) + generated img
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import dnnlib, legacy

REPO = '/home/ygryu/log1/stylegan-xl'
RUN_BASE = os.path.join(REPO, 'training-runs',
                        'imagenet100_128_conv_baseline',
                        '00000-stylegan3-t-imagenet100_128-gpus1-batch32',
                        'network-snapshot.pkl')
RUN_A3   = os.path.join(REPO, 'training-runs',
                        'imagenet100_128_conv_a3',
                        '00000-stylegan3-t-imagenet100_128-gpus1-batch32',
                        'network-snapshot.pkl')

# GPU disabled — force CPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
device = torch.device('cpu')
print(f'device = {device}')


def load_G(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = legacy.load_network_pkl(f)
    G = data['G_ema'].eval().to(device).requires_grad_(False)
    return G


def capture_features(G, z, c):
    feats = {}
    handles = []
    for name in G.synthesis.layer_names:
        layer = getattr(G.synthesis, name)
        def hook(m, inp, out, n=name):
            feats[n] = out.detach().float().cpu().numpy()
        handles.append(layer.register_forward_hook(hook))
    with torch.no_grad():
        img = G(z, c, noise_mode='const')
    for h in handles:
        h.remove()
    return img.detach().cpu().numpy(), feats


print('Loading baseline...')
G_base = load_G(RUN_BASE)
print('Loading a3...')
G_a3   = load_G(RUN_A3)

print(f'z_dim={G_base.z_dim}  c_dim={G_base.c_dim}')
print(f'layer_names (n={len(G_base.synthesis.layer_names)}):')
for n in G_base.synthesis.layer_names:
    print(f'  {n}')
print(f'head_layer_names: {getattr(G_base, "head_layer_names", None)}')

# Fixed latent + class
torch.manual_seed(42)
z = torch.randn(1, G_base.z_dim, device=device)
c = torch.zeros(1, G_base.c_dim, device=device)
c[0, 7] = 1.0   # arbitrary class index
print(f'\nForward passes...')
img_b, feats_b = capture_features(G_base, z, c)
img_a, feats_a = capture_features(G_a3,   z, c)

# normalize generated img to [0,1] for display
def img_to_display(arr):
    a = arr[0]  # (C, H, W)
    a = (a.transpose(1, 2, 0) + 1.0) / 2.0
    return np.clip(a, 0, 1)


layer_names = G_base.synthesis.layer_names
head_set    = set(getattr(G_base, 'head_layer_names', []) or [])
n_layers    = len(layer_names)

# Figure: 2 rows × (n_layers + 1) cols (last col = generated image)
fig, axes = plt.subplots(2, n_layers + 1, figsize=(1.6 * (n_layers + 1), 4.0))

for ci, name in enumerate(layer_names):
    fb = feats_b[name][0]  # (C, H, W)
    fa = feats_a[name][0]
    if fb.ndim != 3:
        continue
    fb_m = fb.mean(axis=0)
    fa_m = fa.mean(axis=0)
    vmin = min(fb_m.min(), fa_m.min())
    vmax = max(fb_m.max(), fa_m.max())
    axes[0, ci].imshow(fb_m, vmin=vmin, vmax=vmax, cmap='viridis')
    axes[1, ci].imshow(fa_m, vmin=vmin, vmax=vmax, cmap='viridis')
    role = 'head' if name in head_set else 'stem'
    title = f'{name}\n({role}, {fb.shape[1]}×{fb.shape[2]})'
    axes[0, ci].set_title(title, fontsize=7)
    for r in (0, 1):
        axes[r, ci].set_xticks([]); axes[r, ci].set_yticks([])

# Last column: the actual generated 128x128 RGB
axes[0, -1].imshow(img_to_display(img_b))
axes[1, -1].imshow(img_to_display(img_a))
axes[0, -1].set_title('generated\n(128×128)', fontsize=7)
for r in (0, 1):
    axes[r, -1].set_xticks([]); axes[r, -1].set_yticks([])

axes[0, 0].set_ylabel('baseline', fontsize=11)
axes[1, 0].set_ylabel('a3',        fontsize=11)

fig.suptitle('Per-layer feature map (channel-mean) — sr128 same latent (z seed=42, class=7)\n'
             'Stem layers: should be identical for baseline (head-only frozen) and a3 except '
             'the last 3 unfrozen.',
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(REPO, 'feature_maps_sr128.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nWrote {out}')

# Also save numeric per-layer difference (L2) for quick scan
print('\nPer-layer L2 difference (baseline vs a3 feature maps):')
print(f'{"layer":15} {"shape":>15}  {"role":>4}  {"L2 diff":>12}  {"rel diff":>9}')
for name in layer_names:
    fb = feats_b[name][0]
    fa = feats_a[name][0]
    if fb.ndim != 3:
        continue
    l2 = float(np.sqrt(((fa - fb) ** 2).sum()))
    norm = float(np.sqrt((fb ** 2).sum()))
    rel = l2 / (norm + 1e-12)
    role = 'head' if name in head_set else 'stem'
    print(f'{name:15} {str(fb.shape):>15}  {role:>4}  {l2:12.4f}  {rel:9.4f}')
