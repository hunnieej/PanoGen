#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, argparse, shutil
import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback

# TODO : Main.py 만들고 자동으로 저장되는거까지 진행
def load_erp_image(path):
    """Load ERP image -> float32 [H,W,3] in [0,1], RGB"""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr  # [H,W,3]

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def erp_to_sphere_pointcloud(img01, step=4, radius=1.0):
    """
    img01: [H,W,3] in [0,1]
    step : sampling stride on ERP grid (>=1)
    """
    H, W = img01.shape[:2]
    vv, uu = np.meshgrid(
        np.arange(0, H, step, dtype=np.float64),
        np.arange(0, W, step, dtype=np.float64),
        indexing='ij'
    )
    # pixel centers
    lon = 2.0 * math.pi * ((uu + 0.5) / W - 0.5)             # [-π, π]
    lat = math.pi/2.0 - math.pi * ((vv + 0.5) / H)           # [-π/2, π/2]

    x = np.cos(lat) * np.cos(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.sin(lon)

    xyz = radius * np.stack([x, y, z], axis=-1).reshape(-1, 3)
    rgb = img01[vv.astype(int), uu.astype(int)].reshape(-1, 3)
    return xyz.astype(np.float32), (rgb * 255.0).astype(np.uint8)

def save_ply_pointcloud(xyz, rgb, path):
    """
    Save ASCII PLY: vertices with per-vertex color.
    xyz: [N,3] float32, rgb: [N,3] uint8
    """
    n = xyz.shape[0]
    header = f"""ply
format ascii 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header)
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

# -------------------------------
# B) UV Sphere Mesh (OBJ + MTL) with ERP texture
# -------------------------------

def generate_uv_sphere(n_lat=180, n_lon=360, radius=1.0, flip_v=True):
    """
    Generate a UV sphere grid with duplicated seam column.
    Returns:
      vertices: [V,3]
      uvs     : [V,2] in [0,1]
      normals : [V,3] (same as unit vertices)
      faces   : [F,3] triangle indices (1-based for OBJ)
    """
    # grid with seam duplication: lon in [0..n_lon], lat in [0..n_lat]
    lon_grid = np.linspace(-math.pi, math.pi, num=n_lon+1, dtype=np.float64)   # includes both ends
    lat_grid = np.linspace( math.pi/2, -math.pi/2, num=n_lat+1, dtype=np.float64)

    # vertex positions
    LON, LAT = np.meshgrid(lon_grid, lat_grid, indexing='xy')  # [n_lat+1, n_lon+1]

    x = np.cos(LAT) * np.cos(LON)
    y = np.sin(LAT)
    z = np.cos(LAT) * np.sin(LON)
    vertices = radius * np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

    # normals (unit)
    normals = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

    # UVs (equirect): u in [0,1], v in [0,1]
    u = (LON / (2.0*math.pi) + 0.5)
    v = (0.5 - LAT / math.pi)
    if flip_v:
        v = 1.0 - v  # many OBJ viewers expect V origin at bottom
    uvs = np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32)

    # faces (two triangles per quad)
    faces = []
    cols = n_lon + 1
    rows = n_lat + 1
    for i in range(rows - 1):
        for j in range(cols - 1):
            a  = i*cols + j
            b  = a + 1
            c  = (i+1)*cols + j
            d  = c + 1
            # winding: counter-clockwise (depends on viewer, this is typical)
            faces.append([a+1, b+1, d+1])  # +1 for OBJ indexing
            faces.append([a+1, d+1, c+1])
    faces = np.asarray(faces, dtype=np.int32)
    return vertices, uvs, normals, faces

def save_obj_with_mtl(vertices, uvs, normals, faces, obj_path, texture_basename, material_name="mat0"):
    """
    Save OBJ referencing an MTL with a diffuse texture (ERP).
    The MTL will be saved alongside OBJ with same stem.
    """
    base_dir = os.path.dirname(obj_path)
    stem     = os.path.splitext(os.path.basename(obj_path))[0]
    mtl_name = stem + ".mtl"
    mtl_path = os.path.join(base_dir, mtl_name)

    # write MTL
    with open(mtl_path, 'w', encoding='utf-8') as f:
        f.write(f"newmtl {material_name}\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write("d 1.0\n")
        f.write("illum 2\n")
        f.write(f"map_Kd {texture_basename}\n")

    # write OBJ
    with open(obj_path, 'w', encoding='utf-8') as f:
        f.write(f"mtllib {mtl_name}\n")
        # vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # UVs
        for t in uvs:
            f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        # normals
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        f.write(f"usemtl {material_name}\n")
        f.write("s off\n")  # no smoothing groups, optional

        # faces (v/vt/vn)
        for tri in faces:
            a, b, c = tri.tolist()
            f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
############################################################
def main():
    ap = argparse.ArgumentParser(description="ERP -> Sphere: PLY pointcloud & UV-sphere OBJ exporter")
    ap.add_argument("--erp", required=True, help="Path to ERP image (e.g., .png, .jpg)")
    ap.add_argument("--outdir", default="out_sphere", help="Output directory")
    ap.add_argument("--ply_step", type=int, default=1, help="Sampling stride for PLY (>=1, smaller = denser)")
    ap.add_argument("--radius", type=float, default=1.0, help="Sphere radius")
    ap.add_argument("--mesh_lat", type=int, default=180, help="Mesh latitudinal segments (rows)")
    ap.add_argument("--mesh_lon", type=int, default=360, help="Mesh longitudinal segments (cols)")
    ap.add_argument("--copy_texture", action="store_true", help="Copy ERP image into outdir for OBJ texture")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    # Load ERP
    img01 = load_erp_image(args.erp)
    H, W = img01.shape[:2]
    print(f"[INFO] Loaded ERP: {args.erp} (H={H}, W={W})")

    # ---- A) Point Cloud (.ply)
    print("[INFO] Building point cloud …")
    xyz, rgb = erp_to_sphere_pointcloud(img01, step=max(1, args.ply_step), radius=args.radius)
    ply_path = os.path.join(args.outdir, "erp_sphere_points.ply")
    save_ply_pointcloud(xyz, rgb, ply_path)
    print(f"[OK] Saved PLY: {ply_path} (N={xyz.shape[0]} points)")

    # ---- B) UV Sphere Mesh (.obj + .mtl) with ERP texture
    print("[INFO] Building UV sphere mesh …")
    V, T, N, F = generate_uv_sphere(n_lat=args.mesh_lat, n_lon=args.mesh_lon, radius=args.radius, flip_v=True)

    obj_path = os.path.join(args.outdir, "erp_textured_sphere.obj")
    # texture path handling
    tex_basename = os.path.basename(args.erp)
    if args.copy_texture:
        dst_tex = os.path.join(args.outdir, tex_basename)
        if os.path.abspath(dst_tex) != os.path.abspath(args.erp):
            shutil.copy2(args.erp, dst_tex)
        print(f"[OK] Copied texture into outdir: {dst_tex}")

    save_obj_with_mtl(V, T, N, F, obj_path, texture_basename=tex_basename, material_name="erp_tex")
    print(f"[OK] Saved OBJ+MTL: {obj_path} (+ .mtl)")

    print("\n[READY]")
    print(" - Open the PLY in Meshlab/CloudCompare to see colored points on a sphere.")
    print(" - Open the OBJ in Blender/Meshlab; make sure the ERP image is in the same folder as the OBJ (or use --copy_texture).")

if __name__ == "__main__":
    main()
