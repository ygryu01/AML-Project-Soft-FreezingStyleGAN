#!/usr/bin/env python
"""Compute sr256 (128->256) drift+spectrum and merge into analysis_results.json.

sr256 was skipped by analyze_drift_spectrum.py because its stem reference
pretrained/imagenet128.pkl was missing. With that pkl now present, this script
analyzes the 4 converged 256 runs against it and appends a 'sr256' block,
reusing the exact same metric code (load_G / analyze_run) as the main script.

Variant dir suffixes (baseline, variant_a, variant_a2, variant_a3) are
normalized to the JSON's convention (baseline, a, a2, a3).
"""
import os, json, shutil, sys, types

# Some 256 snapshots (variant_a2/a3) were saved with a NEWER timm, where the
# discriminator's DeiT feature net lives in `timm.models.deit` and its building
# blocks in `timm.layers`. The current env (timm 0.4.12) has neither path. We
# never use the discriminator's feature network for drift analysis (only
# G_ema.synthesis), so install a *fallback* meta-path finder that fabricates a
# dummy module (any attribute -> throwaway class) for timm.* paths that don't
# otherwise exist. Real timm modules (e.g. vision_transformer) still load
# normally because this finder is appended AFTER the standard finders.
import importlib.abc, importlib.machinery

class _DummyBase:
    # Tolerate any constructor signature (e.g. enums like Format('NCHW')) and
    # any pickled state — these objects are reconstructed only to let the D
    # state deserialize; they are never used.
    def __new__(cls, *a, **k):
        return object.__new__(cls)
    def __init__(self, *a, **k):
        pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

def _dummy_class(name):
    return type(name, (_DummyBase,), {})

class _DummyTimmFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith('timm.'):
            return None
        return importlib.machinery.ModuleSpec(fullname, self)
    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__getattr__ = _dummy_class  # PEP 562 module getattr -> permissive class
        m.__path__ = []  # treat as package so submodules resolve
        return m
    def exec_module(self, module):
        pass

if not any(isinstance(f, _DummyTimmFinder) for f in sys.meta_path):
    sys.meta_path.append(_DummyTimmFinder())

import analyze_drift_spectrum as A

STEM256 = os.path.join(A.REPO, 'pretrained', 'imagenet128.pkl')
RUN_MAP = [
    ('baseline',  'imagenet100_256_converged_baseline'),
    ('a',         'imagenet100_256_converged_variant_a'),
    ('a2',        'imagenet100_256_converged_variant_a2'),
    ('a3',        'imagenet100_256_converged_variant_a3'),
]

def main():
    assert os.path.isfile(STEM256), f'missing stem: {STEM256}'
    out_json = os.path.join(A.REPO, 'analysis_results.json')
    results = json.load(open(out_json))

    G_stem = A.load_G(STEM256)
    sr256 = {}
    for var, rd in RUN_MAP:
        snap = A._find_snap(os.path.join(A.RUNDIR, rd))
        if snap is None:
            print(f'[sr256 {var}] no snapshot -> skip'); continue
        print(f'[sr256 {var}] loading {snap}', flush=True)
        try:
            G_tr = A.load_G(snap)
        except Exception as e:
            print(f'   FAILED to load ({type(e).__name__}: {e}) -> skip '
                  f'[snapshot is likely truncated/corrupt]', flush=True)
            continue
        per_layer = A.analyze_run(G_stem, G_tr)
        sr256[var] = per_layer
        cd = [m['rel_drift'] for k, m in per_layer.items() if 'affine' not in k]
        mean_d = sum(cd) / max(len(cd), 1)
        max_d = max(cd) if cd else 0
        print(f'   layers={len(per_layer)}  mean_conv_drift={mean_d:.4f}  max={max_d:.4f}', flush=True)

    if not sr256:
        print('No sr256 runs analyzed; nothing written.'); return

    bak = out_json + '.bak'
    if not os.path.isfile(bak):
        shutil.copy2(out_json, bak)
        print(f'Backed up original -> {bak}')
    results['sr256'] = sr256
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Merged sr256 ({list(sr256)}) into {out_json}')

if __name__ == '__main__':
    main()
