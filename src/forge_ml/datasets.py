"""Small bounded datasets, with fixed stratified splits and no learned preprocessing."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

MAX_ROWS = 4096

def stratified_split(y, seed=2026):
    rng = np.random.default_rng(seed)
    splits = [[], [], []]
    for c in np.unique(y):
        ids = rng.permutation(np.flatnonzero(y == c))
        a, b = max(1, int(len(ids) * .7)), max(1, int(len(ids) * .15))
        for out, part in zip(splits, (ids[:a], ids[a:a+b], ids[a+b:])):
            out.extend(part.tolist())
    return {k: rng.permutation(v).tolist() for k, v in zip(('train', 'validation', 'test'), splits)}


def dataset_2d(name='moons', n=240, noise=.12, seed=42, points=None):
    if name not in ('xor', 'moons', 'spiral', 'custom'):
        raise ValueError('Unknown 2D dataset')
    if type(n) is not int or not (6 if name=='custom' else 20) <= n <= 2048 or not 0 <= noise <= .8:
        raise ValueError('Require 20–2048 rows (custom: 6–2048) and noise in [0, 0.8]')
    rng = np.random.default_rng(seed)
    if name == 'custom':
        if not isinstance(points, list) or not 6 <= len(points) <= 2048:
            raise ValueError('Custom data requires 6–2048 points')
        if any(not isinstance(p, dict) or set(p) != {'x','y','label'} for p in points):
            raise ValueError('Custom points must contain x, y and label')
        if any(type(p[k]) not in (int,float) for p in points for k in ('x','y')):
            raise ValueError('Custom coordinates must be numbers')
        x = np.asarray([[p['x'], p['y']] for p in points], dtype=np.float32)
        raw_y = [p['label'] for p in points]
        if any(type(v) is not int or v not in (0, 1) for v in raw_y):
            raise ValueError('Custom labels must be integer 0 or 1')
        y = np.asarray(raw_y, dtype=np.int32)
        if not np.isfinite(x).all() or np.abs(x).max() > 5:
            raise ValueError('Custom coordinates must be finite in [-5,5]')
        if min(np.bincount(y, minlength=2)) < 3:
            raise ValueError('Each custom class needs at least three points')
    else:
        y = (np.arange(n) % 2).astype(np.int32)
        if name == 'xor':
            signs = rng.choice([-1,1], size=(n,2))
            x = signs * rng.uniform(.25, 1.25, size=(n,2))
            y = (signs[:,0] * signs[:,1] > 0).astype(np.int32)
        elif name == 'moons':
            theta = rng.uniform(0, np.pi, n)
            x = np.column_stack((np.where(y == 0, np.cos(theta), 1 - np.cos(theta)),
                                 np.where(y == 0, np.sin(theta), .5 - np.sin(theta))))
            x -= [.5,.25]
            x *= 1.2
        else:
            radius = rng.uniform(.08, 1.7, n)
            theta = radius * 3.8 + y * np.pi
            x = np.column_stack((radius * np.sin(theta), radius * np.cos(theta)))
        x = (x + rng.normal(0, noise, (n,2))).astype(np.float32)
    split = stratified_split(y, seed + 1000)
    identity = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    return {'x': x, 'y': y, 'splits': split, 'id': name, 'checksum': identity}


def load_digits_asset(source):
    obj = json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else source
    if not isinstance(obj, dict) or obj.get('version') != 1 or obj.get('rows') != 1797:
        raise ValueError('Unsupported digits asset')
    pixels, labels = obj.get('pixels'), obj.get('labels')
    if not isinstance(pixels, list) or len(pixels) != 1797 * 64 or not isinstance(labels,list) or len(labels) != 1797:
        raise ValueError('Digits data size mismatch')
    if any(type(v) is not int or not 0 <= v <= 16 for v in pixels):
        raise ValueError('Invalid pixels')
    if any(type(v) is not int or not 0 <= v < 10 for v in labels):
        raise ValueError('Invalid class labels')
    raw = np.array(pixels, dtype=np.uint8)
    y = np.array(labels, dtype=np.int32)
    digest = hashlib.sha256(raw.tobytes() + y.astype('<i4').tobytes()).hexdigest()
    if digest != obj.get('checksum'):
        raise ValueError('Digits checksum mismatch')
    split = obj.get('splits')
    if not isinstance(split,dict) or set(split)!={'train','validation','test'} or any(not isinstance(v,list) for v in split.values()):
        raise ValueError('Invalid digits split structure')
    ids = sum([split[k] for k in ('train','validation','test')], [])
    if any(type(v) is not int for v in ids) or sorted(ids) != list(range(1797)):
        raise ValueError('Digits split must partition rows exactly')
    if split != stratified_split(y,2026):
        raise ValueError('Digits split must match the fixed seed-2026 stratified partition')
    if not isinstance(obj.get('provenance'),dict):raise ValueError('Digits provenance is required')
    # Fixed physical range normalization; no fitted statistics and no test leakage.
    return {'x': raw.reshape(1797,64).astype(np.float32) / np.float32(16), 'y': y,
            'splits': split, 'id':'digits', 'checksum': digest, 'provenance':obj['provenance']}
