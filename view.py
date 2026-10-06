#!/usr/bin/env python3
"""View a voxel .npz as a solid mesh (PyVista).

Usage:
    python view.py output.npz
    python view.py samples.npz --index 3 --level 0.5
    python view.py output.npz --save mesh.ply
"""
import argparse
import sys

import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes


def main():
    p = argparse.ArgumentParser(description="View voxel .npz as solid geometry")
    p.add_argument("file")
    p.add_argument("--key", default=None, help="array key (default: first key)")
    p.add_argument("--index", type=int, default=0, help="sample index if batched")
    p.add_argument("--level", type=float, default=None,
                   help="iso-level (default: 0 if data has negatives, else 0.5)")
    p.add_argument("--save", default=None, help="export mesh (.ply/.stl/.vtk)")
    args = p.parse_args()

    data = np.load(args.file)
    key = args.key or list(data.keys())[0]
    if key not in data:
        sys.exit(f"Key '{key}' not found. Available: {list(data.keys())}")

    vol = np.asarray(data[key], dtype=np.float32)
    print(f"[{key}] shape={vol.shape} min={vol.min():.3f} max={vol.max():.3f}")

    if vol.ndim == 5:                       # (N, C, D, H, W)
        vol = vol[args.index]
    if vol.ndim == 4:                       # (C, D, H, W) or (N, D, H, W)
        vol = vol[0] if vol.shape[0] == 1 else vol[args.index]
    if vol.ndim != 3:
        sys.exit(f"Can't reduce shape {vol.shape} to a 3D volume")

    level = args.level
    if level is None:
        level = 0.0 if vol.min() < 0 else 0.5

    if not (vol.min() < level < vol.max()):
        sys.exit(f"Level {level} outside data range [{vol.min():.3f}, {vol.max():.3f}]. "
                 f"Pass --level.")

    # pad with background so the surface closes at the grid borders
    vol = np.pad(vol, 1, constant_values=vol.min())

    verts, faces, _, _ = marching_cubes(vol, level=level)
    print(f"mesh: {len(verts)} vertices, {len(faces)} faces", flush=True)

    faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    mesh = pv.PolyData(np.ascontiguousarray(verts, dtype=np.float64), faces_pv)

    if args.save:
        mesh.save(args.save)
        print(f"saved {args.save}")

    mesh.plot(color="lightsteelblue", smooth_shading=True, show_edges=False)


if __name__ == "__main__":
    main()