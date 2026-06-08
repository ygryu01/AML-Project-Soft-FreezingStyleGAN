#!/usr/bin/env python
"""LPIPS pair-wise distances for sr128 counterfactual injection.

For each variant v ∈ {a, a2, a3, c, d_cosine, d_linear}, over N random
(z, class) samples:

    img_b   = G_baseline(z, c)                 -- baseline normal
    img_v   = G_v(z, c)                        -- variant normal
    img_cf  = G_v(z, c | inject baseline L18) -- counterfactual H_v(S_0(z))

    d_swap[v] = LPIPS(img_v,  img_cf)   -- how much does stem swap change output?
    d_var[v]  = LPIPS(img_v,  img_b)    -- how different is variant from baseline?

Reports mean ± std per variant. Outputs a JSON summary.
"""
import os, sys, json, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

import numpy as np, torch
import dnnlib, legacy
import lpips as lp

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

VARIANTS_ALL = [
    ('baseline', 'imagenet100_128_conv_baseline'),
    ('a',        'imagenet100_128_conv_a'),
    ('a2',       'imagenet100_128_conv_a2'),
    ('a3',       'imagenet100_128_conv_a3'),
    ('c',        'imagenet100_128_conv_c'),
    ('d_cosine', 'imagenet100_128_variant_d_cosine'),
    ('d_linear', 'imagenet100_128_variant_d_linear'),
]
BOUNDARY = 'L18_84_1024'

N_SAMPLES = int(os.environ.get('N_SAMPLES', '1000'))
BATCH = int(os.environ.get('BATCH', '8'))
SEED_OFFSET = 0


def load_G(path):
    with open(path, 'rb') as f:
        return legacy.load_network_pkl(f)['G_ema'].eval().to(device).requires_grad_(False)


# Load all G
print(f'Loading {len(VARIANTS_ALL)} G_ema...')
Gs = {v: load_G(os.path.join(RUNDIR, d, SUBDIR, 'network-snapshot.pkl'))
      for v, d in VARIANTS_ALL}
G0 = Gs['baseline']
z_dim, c_dim = G0.z_dim, G0.c_dim
print(f'  z_dim={z_dim}  c_dim={c_dim}')

print('Loading LPIPS (AlexNet)...')
lpips_model = lp.LPIPS(net='alex').to(device).eval()
for p in lpips_model.parameters():
    p.requires_grad_(False)


def forward_capture(G, z, c, layer_name):
    captured = {}
    layer = getattr(G.synthesis, layer_name)
    def hook(m, inp, out):
        captured['out'] = out.detach().clone()
    h = layer.register_forward_hook(hook)
    with torch.no_grad():
        img = G(z, c, noise_mode='const')
    h.remove()
    return img, captured['out']


def forward_inject(G, z, c, layer_name, replacement):
    layer = getattr(G.synthesis, layer_name)
    def hook(m, inp, out):
        return replacement
    h = layer.register_forward_hook(hook)
    with torch.no_grad():
        img = G(z, c, noise_mode='const')
    h.remove()
    return img


print(f'\nRunning {N_SAMPLES} samples, batch={BATCH}, boundary={BOUNDARY}')

# Pre-generate (seed-based) z and random class
rng = np.random.RandomState(0)
class_idx_seq = rng.randint(0, c_dim, size=N_SAMPLES)

results = {v: {'d_swap': [], 'd_var': [], 'd_head': []} for v, _ in VARIANTS_ALL[1:]}

n_done = 0
while n_done < N_SAMPLES:
    bs = min(BATCH, N_SAMPLES - n_done)
    # Build batch z and class
    z_batch_list = []
    for i in range(bs):
        torch.manual_seed(SEED_OFFSET + n_done + i)
        z_batch_list.append(torch.randn(1, z_dim))
    z_batch = torch.cat(z_batch_list, dim=0).to(device)
    c_batch = torch.zeros(bs, c_dim, device=device)
    for i in range(bs):
        c_batch[i, int(class_idx_seq[n_done + i])] = 1.0

    # baseline forward + capture
    img_b, feat0 = forward_capture(G0, z_batch, c_batch, BOUNDARY)

    for v, _ in VARIANTS_ALL[1:]:
        img_v, _ = forward_capture(Gs[v], z_batch, c_batch, BOUNDARY)
        img_cf = forward_inject(Gs[v], z_batch, c_batch, BOUNDARY, feat0)
        with torch.no_grad():
            d_sw = lpips_model(img_v,  img_cf).view(-1).cpu().numpy().tolist()
            d_va = lpips_model(img_v,  img_b ).view(-1).cpu().numpy().tolist()
            d_he = lpips_model(img_cf, img_b ).view(-1).cpu().numpy().tolist()
        results[v]['d_swap'].extend(d_sw)
        results[v]['d_var'].extend(d_va)
        results[v]['d_head'].extend(d_he)

    n_done += bs
    if n_done % 100 == 0 or n_done == N_SAMPLES:
        print(f'  {n_done}/{N_SAMPLES}')

# Summarize
print('\n=== LPIPS pair-wise (sr128 counterfactual @ L18_84_1024) ===')
print(f'{"variant":10}  {"d_swap":>12}  {"d_var":>12}  {"d_head":>12}  '
      f'{"swap/var":>9}  {"head/var":>9}')
print('-' * 78)
summary = {}
for v, _ in VARIANTS_ALL[1:]:
    sw = np.array(results[v]['d_swap'])
    va = np.array(results[v]['d_var'])
    he = np.array(results[v]['d_head'])
    sw_m, sw_s = float(sw.mean()), float(sw.std())
    va_m, va_s = float(va.mean()), float(va.std())
    he_m, he_s = float(he.mean()), float(he.std())
    summary[v] = {
        'd_swap_mean': sw_m, 'd_swap_std': sw_s,
        'd_var_mean':  va_m, 'd_var_std':  va_s,
        'd_head_mean': he_m, 'd_head_std': he_s,
        'ratio_swap_over_var': sw_m / max(va_m, 1e-12),
        'ratio_head_over_var': he_m / max(va_m, 1e-12),
        'n': len(sw),
    }
    print(f'{v:10}  {sw_m:6.4f}±{sw_s:.3f}  {va_m:6.4f}±{va_s:.3f}  '
          f'{he_m:6.4f}±{he_s:.3f}  '
          f'{summary[v]["ratio_swap_over_var"]:9.3f}  '
          f'{summary[v]["ratio_head_over_var"]:9.3f}')

out_json = os.path.join(REPO, 'lpips_counterfactual_sr128.json')
with open(out_json, 'w') as f:
    json.dump({'samples': N_SAMPLES, 'boundary': BOUNDARY, 'per_variant': summary}, f, indent=2)
print(f'\nWrote {out_json}')
