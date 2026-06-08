#!/usr/bin/env python
"""Generalized feature-map visualization across all ImageNet super-res stages.

For each stage where baseline is available, runs baseline + all sibling
variants through the same fixed (z, class) forward pass on CPU, captures
every synthesis layer output, and produces one PNG per stage in the same
format as feature_map_sr128.py:

  rows = {baseline, variant_1, variant_2, ...}
  cols = synthesis layer (channel-mean feature map) + final generated image

Stages covered:
  sr64  — baseline, a, a2, a3, b, c
  sr128 — baseline, a, a2, a3, c, d_cosine, d_linear
  sr256 — baseline, a, a2, a3   (head_layers=7, runs labeled 'variant_*')

sr32 is skipped (no baseline on this machine).
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Some snapshots (sr256 from elicer machine) were pickled against newer timm
# (1.x) where modules like timm.layers / timm.models.deit exist. Our env has
# timm 0.4.12 only. We only need G_ema (not D/backbone) — so we monkey-patch
# legacy.LegacyUnpickler.find_class to return a generic stub class for any
# missing module/attribute.
class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

import numpy as np
import torch
import matplotlib.pyplot as plt
import dnnlib, legacy

_orig_find_class = legacy._LegacyUnpickler.find_class
def _patched_find_class(self, module, name):
    try:
        return _orig_find_class(self, module, name)
    except (ModuleNotFoundError, ImportError, AttributeError) as e:
        print(f'  [stub] {module}.{name} ({type(e).__name__})')
        return _Stub
legacy._LegacyUnpickler.find_class = _patched_find_class

REPO = '/home/ygryu/log1/stylegan-xl'
RUNDIR = os.path.join(REPO, 'training-runs')

os.environ['CUDA_VISIBLE_DEVICES'] = ''
device = torch.device('cpu')
print(f'device = {device}')

SEED = 42
CLASS_IDX = 7   # any fixed class

# Stage -> ordered (variant_label, run_dir_name, snapshot_subdir)
STAGES = {
    'sr64': [
        ('baseline', 'imagenet100_64_conv_baseline',
         '00000-stylegan3-t-imagenet100_64-gpus1-batch32'),
        ('a',        'imagenet100_64_conv_a',
         '00000-stylegan3-t-imagenet100_64-gpus1-batch32'),
        ('a2',       'imagenet100_64_conv_a2',
         '00000-stylegan3-t-imagenet100_64-gpus1-batch32'),
        ('a3',       'imagenet100_64_conv_a3',
         '00000-stylegan3-t-imagenet100_64-gpus1-batch32'),
        ('b',        'imagenet100_64_conv_b',
         '00000-stylegan3-t-imagenet100_64-gpus1-batch32'),
        ('c',        'imagenet100_64_conv_c',
         '00000-stylegan3-t-imagenet100_64-gpus1-batch32'),
    ],
    'sr128': [
        ('baseline', 'imagenet100_128_conv_baseline',
         '00000-stylegan3-t-imagenet100_128-gpus1-batch32'),
        ('a',        'imagenet100_128_conv_a',
         '00000-stylegan3-t-imagenet100_128-gpus1-batch32'),
        ('a2',       'imagenet100_128_conv_a2',
         '00000-stylegan3-t-imagenet100_128-gpus1-batch32'),
        ('a3',       'imagenet100_128_conv_a3',
         '00000-stylegan3-t-imagenet100_128-gpus1-batch32'),
        ('c',        'imagenet100_128_conv_c',
         '00000-stylegan3-t-imagenet100_128-gpus1-batch32'),
        ('d_cosine', 'imagenet100_128_variant_d_cosine',
         '00000-stylegan3-t-imagenet100_128-gpus1-batch32'),
        ('d_linear', 'imagenet100_128_variant_d_linear',
         '00000-stylegan3-t-imagenet100_128-gpus1-batch32'),
    ],
    'sr256': [
        ('baseline', 'imagenet100_256_converged_baseline',
         '00000-stylegan3-t-imagenet100_256-gpus1-batch32'),
        ('a',        'imagenet100_256_converged_variant_a',
         '00000-stylegan3-t-imagenet100_256-gpus1-batch32'),
        ('a2',       'imagenet100_256_converged_variant_a2',
         '00000-stylegan3-t-imagenet100_256-gpus1-batch32'),
        ('a3',       'imagenet100_256_converged_variant_a3',
         '00000-stylegan3-t-imagenet100_256-gpus1-batch32'),
    ],
}


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


def img_to_display(arr):
    a = arr[0]
    a = (a.transpose(1, 2, 0) + 1.0) / 2.0
    return np.clip(a, 0, 1)


def run_stage(stage_name, runs):
    # 1) verify all snapshots present
    snaps = []
    for v, d, sub in runs:
        p = os.path.join(RUNDIR, d, sub, 'network-snapshot.pkl')
        if not os.path.isfile(p):
            print(f'[{stage_name}/{v}] snapshot MISSING — skipping this variant')
            continue
        snaps.append((v, p))
    if not snaps or snaps[0][0] != 'baseline':
        print(f'[{stage_name}] no baseline → skip whole stage')
        return

    print(f'\n========== {stage_name} : {len(snaps)} variants ==========')
    # 2) load first G to set z_dim / c_dim
    print(f'[{stage_name}] loading baseline...')
    G0 = load_G(snaps[0][1])
    torch.manual_seed(SEED)
    z = torch.randn(1, G0.z_dim, device=device)
    c = torch.zeros(1, G0.c_dim, device=device)
    c[0, CLASS_IDX] = 1.0
    img_base, feats_base = capture_features(G0, z, c)
    head_set = set(getattr(G0, 'head_layer_names', []) or [])
    layer_names = list(G0.synthesis.layer_names)
    print(f'   layers={len(layer_names)}, head={list(head_set)[:3]}…')

    all_feats = {'baseline': feats_base}
    all_imgs  = {'baseline': img_base}

    ok_variants = [snaps[0]]
    for v, p in snaps[1:]:
        print(f'[{stage_name}] loading {v}...')
        try:
            G = load_G(p)
        except Exception as e:
            print(f'   [skip] {v} load failed: {type(e).__name__}: {e}')
            continue
        try:
            img_v, feats_v = capture_features(G, z, c)
        except Exception as e:
            print(f'   [skip] {v} forward failed: {type(e).__name__}: {e}')
            del G
            continue
        all_feats[v] = feats_v
        all_imgs[v]  = img_v
        ok_variants.append((v, p))
        del G
    snaps = ok_variants

    # 3) plot: n_variants rows × (n_layers + 1) cols
    n_v = len(snaps)
    n_l = len(layer_names)
    fig, axes = plt.subplots(n_v, n_l + 1, figsize=(1.45 * (n_l + 1), 1.7 * n_v))
    if n_v == 1:
        axes = np.array([axes])

    # for each layer column compute shared vmin/vmax across all variants
    for ci, name in enumerate(layer_names):
        layer_panels = []
        for v, _ in snaps:
            f = all_feats[v][name][0]
            if f.ndim != 3:
                layer_panels.append(None)
                continue
            layer_panels.append(f.mean(axis=0))
        non_none = [p for p in layer_panels if p is not None]
        if not non_none:
            continue
        vmin = min(p.min() for p in non_none)
        vmax = max(p.max() for p in non_none)
        for ri, (v, _) in enumerate(snaps):
            ax = axes[ri, ci]
            p = layer_panels[ri]
            if p is None:
                ax.set_visible(False)
            else:
                ax.imshow(p, vmin=vmin, vmax=vmax, cmap='viridis')
            ax.set_xticks([]); ax.set_yticks([])
        role = 'head' if name in head_set else 'stem'
        shape0 = all_feats[snaps[0][0]][name][0].shape
        axes[0, ci].set_title(f'{name}\n({role}, {shape0[1]}×{shape0[2]})', fontsize=6)

    # Last column: generated images
    for ri, (v, _) in enumerate(snaps):
        ax = axes[ri, -1]
        ax.imshow(img_to_display(all_imgs[v]))
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, -1].set_title('generated', fontsize=7)

    # variant row labels (left)
    for ri, (v, _) in enumerate(snaps):
        axes[ri, 0].set_ylabel(v, fontsize=10)

    fig.suptitle(f'{stage_name}: per-layer channel-mean feature map vs baseline '
                 f'(z seed={SEED}, class={CLASS_IDX})', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(REPO, f'feature_maps_{stage_name}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'   wrote {out}')

    # numeric per-layer L2 vs baseline
    print(f'\n   Per-layer L2 difference vs baseline:')
    print(f'   {"layer":15} {"role":>4}  ' + '  '.join(f'{v:>9}' for v in [s[0] for s in snaps[1:]]))
    for name in layer_names:
        fb = all_feats['baseline'][name][0]
        if fb.ndim != 3:
            continue
        norm = float(np.sqrt((fb ** 2).sum()))
        diffs = []
        for v, _ in snaps[1:]:
            fv = all_feats[v][name][0]
            l2 = float(np.sqrt(((fv - fb) ** 2).sum()))
            diffs.append(l2 / (norm + 1e-12))
        role = 'head' if name in head_set else 'stem'
        print(f'   {name:15} {role:>4}  ' + '  '.join(f'{x:9.4f}' for x in diffs))


_only = os.environ.get('STAGES_ONLY', '').split(',')
_only = [x.strip() for x in _only if x.strip()]
for stage_name, runs in STAGES.items():
    if _only and stage_name not in _only:
        continue
    run_stage(stage_name, runs)

print('\nAll done.')
