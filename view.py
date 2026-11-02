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
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

def create_mesh(vol, level=None, sigma=1.0, smooth_iters=30, keep_largest=True):
    vol = np.asarray(vol, dtype=np.float32)

    if level is None:
        level = 0.0 if vol.min() < 0 else 0.5

    if not (vol.min() < level < vol.max()):
        raise ValueError(
            f"Level {level} outside data range "
            f"[{vol.min():.3f}, {vol.max():.3f}]. Pass --level."
        )

    # Pad so the surface closes at the borders
    vol = np.pad(vol, 2, constant_values=vol.min())

    # Blur the field -> removes staircase + isolated noisy voxels
    if sigma > 0:
        vol = gaussian_filter(vol, sigma=sigma)
        if not (vol.min() < level < vol.max()):
            raise ValueError("Level fell outside range after blur; lower --sigma or --level.")

    verts, faces, _, _ = marching_cubes(vol, level=level)

    faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    mesh = pv.PolyData(np.ascontiguousarray(verts, dtype=np.float64), faces_pv)

    # Drop floating specks (common in GAN output)
    if keep_largest:
        mesh = mesh.connectivity(extraction_mode="largest")
        mesh = mesh.extract_surface()

    # Taubin smoothing: smooths without shrinking the shape
    if smooth_iters > 0:
        mesh = mesh.smooth_taubin(n_iter=smooth_iters, pass_band=0.05)

    mesh = mesh.compute_normals(auto_orient_normals=True, split_vertices=False)

    print(f"mesh: {mesh.n_points} vertices, {mesh.n_cells} faces", flush=True)
    return mesh
    
def main():
    p = argparse.ArgumentParser(description="View voxel .npz as solid geometry")
    p.add_argument("file")
    p.add_argument("--key", default=None, help="array key (default: first key)")
    p.add_argument("--index", type=int, default=0, help="sample index if batched")
    p.add_argument(
        "--level",
        type=float,
        default=None,
        help="iso-level (default: 0 if data has negatives, else 0.5)"
    )
    p.add_argument("--save", default=None, help="export mesh (.ply/.stl/.vtk)")
    p.add_argument("--sigma", type=float, default=1.0, help="gaussian blur on volume (0=off)")
    p.add_argument("--smooth", type=int, default=30, help="Taubin smoothing iterations (0=off)")
    p.add_argument("--keep-all", action="store_true", help="keep small disconnected pieces")
    
    args = p.parse_args()

    data = np.load(args.file)
    key = args.key or list(data.keys())[0]

    if key not in data:
        sys.exit(f"Key '{key}' not found. Available: {list(data.keys())}")

    vol = np.asarray(data[key], dtype=np.float32)

    print(
        f"[{key}] shape={vol.shape} "
        f"min={vol.min():.3f} max={vol.max():.3f}"
    )

    if vol.ndim == 5:                       # (N, C, D, H, W)
        vol = vol[args.index]

    if vol.ndim == 4:                       # (C, D, H, W) or (N, D, H, W)
        vol = vol[0] if vol.shape[0] == 1 else vol[args.index]

    if vol.ndim != 3:
        sys.exit(f"Can't reduce shape {vol.shape} to a 3D volume")

    try:
        mesh = create_mesh(vol, args.level, args.sigma, args.smooth, not args.keep_all)
    except ValueError as e:
        sys.exit(str(e))

    if args.save:
        mesh.save(args.save)
        print(f"saved {args.save}")

    mesh.plot(
        color="lightsteelblue",
        smooth_shading=True,
        show_edges=False
    )

if __name__ == "__main__":
    main()