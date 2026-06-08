#!/usr/bin/env python
"""Build imagenet100_256.zip from bhomoon's ImageNet validation set.

For each of the 100 class labels used in our existing imagenet100_128.zip,
take all images from the corresponding synset directory in val/, center-
crop + resize to 256x256, and save into a flat zip with the same class
folder naming ('00001/img_NN_NNNNN.png', etc.) so downstream code matches.

Result: ~5000 real images (100 class x 50/class) for FID reference.
"""
import os, sys, json, io, zipfile
from pathlib import Path
from PIL import Image, ImageOps

VAL_ROOT = Path('/data3/bhomoon/ImageNet/ILSVRC/Data/CLS-LOC/val')
REF_ZIP  = '/home/ygryu/log1/stylegan-xl/data/imagenet100_128.zip'
DEST_ZIP = '/home/ygryu/log1/stylegan-xl/data/imagenet100_256.zip'
RES = 256


def center_crop_resize(img, res):
    img = ImageOps.exif_transpose(img).convert('RGB')
    w, h = img.size
    crop = min(w, h)
    left = (w - crop) // 2
    top  = (h - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))
    return img.resize((res, res), Image.LANCZOS)


# 1) extract the 100 class labels from our existing 128 zip
print(f'Reading reference dataset: {REF_ZIP}')
with zipfile.ZipFile(REF_ZIP) as z:
    ref_meta = json.load(z.open('dataset.json'))
class_labels = sorted(set(int(l[0].split('/')[0]) for l in ref_meta['labels']))
print(f'100 class labels (first 10): {class_labels[:10]}')
print(f'100 class labels (last 5):   {class_labels[-5:]}')
assert len(class_labels) == 100

# 2) build label-index → synset id mapping (alphabetical val/ dir = ImageNet std)
val_synsets = sorted(os.listdir(VAL_ROOT))
assert len(val_synsets) == 1000, f'expected 1000 synsets, got {len(val_synsets)}'
print(f'val[0]={val_synsets[0]}  val[1]={val_synsets[1]}  val[999]={val_synsets[999]}')

selected = [(label, val_synsets[label]) for label in class_labels]
print(f'first selected: label={selected[0][0]} synset={selected[0][1]}')
print(f'last  selected: label={selected[-1][0]} synset={selected[-1][1]}')

# 3) iterate, resize, write to zip
new_labels = []
total = 0
print(f'\nWriting {DEST_ZIP} ...')
with zipfile.ZipFile(DEST_ZIP, 'w', zipfile.ZIP_STORED) as outz:
    for ci, (label, synset) in enumerate(selected):
        folder = f'{label:05d}'
        syn_dir = VAL_ROOT / synset
        files = sorted(os.listdir(syn_dir))
        for i, fn in enumerate(files):
            src = syn_dir / fn
            try:
                with Image.open(src) as im:
                    out_img = center_crop_resize(im, RES)
            except Exception as e:
                print(f'  skip {src}: {e}')
                continue
            buf = io.BytesIO()
            out_img.save(buf, format='PNG', optimize=False)
            out_name = f'{folder}/img_{i:03d}_{total:05d}.png'
            outz.writestr(out_name, buf.getvalue())
            new_labels.append([out_name, label])
            total += 1
        if (ci + 1) % 10 == 0:
            print(f'  [{ci+1}/100] last={folder} ({synset}) total_imgs={total}')
    outz.writestr('dataset.json', json.dumps({'labels': new_labels}))

print(f'\nDone. {total} images, {DEST_ZIP}')
sz = os.path.getsize(DEST_ZIP) / 1024 / 1024
print(f'zip size: {sz:.1f} MB')
