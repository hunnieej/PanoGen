# -*- coding: utf-8 -*-
# ERP 시각화: Fibonacci lattice + 89-view footprints (FOV 80°)
# - 축을 각도(경도 0~360°, 위도 90°→-90°)
# - look-at basis로 타일 중심과 프러스텀 정확히 정렬
# - 타일마다 "중심 X"와 "테두리 4변" 동일 색 (팔레트 사전으로 순서와 무관하게 매칭)
# - 합본/링별 저장 지원

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --------------------------- Sphere sampling ---------------------------

def fibonacci_sphere(n_points: int):
    i = np.arange(n_points, dtype=np.float64)
    y = 1.0 - 2.0 * (i + 0.5) / n_points
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * i
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    return x, y, z

def sphere_to_erp(x, y, z):
    lam = np.arctan2(x, z)                 # [-π, π]
    phi = np.arcsin(np.clip(y, -1.0, 1.0)) # [-π/2, π/2]
    u = (lam + np.pi) / (2.0 * np.pi)      # [0,1]
    v = (np.pi/2.0 - phi) / np.pi          # [0,1] (top=North)
    return u, v

# ----------------------------- View rig --------------------------------

def ring(pitch_deg: float, n: int):
    return [(yaw, pitch_deg) for yaw in np.linspace(0.0, 360.0, num=n, endpoint=False)]

def make_89_views():
    views = []
    views += ring(+90.0, 4)
    views += ring(+77.5, 8)
    views += ring(+45.0, 11)
    views += ring(+22.5, 14)
    views += ring(0.0, 15)
    views += ring(-22.5, 14)
    views += ring(-45.0, 11)
    views += ring(-77.5, 8)
    views += ring(-90.0, 4)
    return views

# --------------------------- Orientation -------------------------------

def dir_from_yaw_pitch(yaw_deg, pitch_deg):
    lam = math.radians(yaw_deg); phi = math.radians(pitch_deg)
    x = math.cos(phi) * math.sin(lam)
    y = math.sin(phi)
    z = math.cos(phi) * math.cos(lam)
    return np.array([x, y, z], dtype=np.float64)

def look_at_basis(yaw_deg, pitch_deg):
    fwd = dir_from_yaw_pitch(yaw_deg, pitch_deg); fwd /= np.linalg.norm(fwd)
    up_ref = np.array([0,0,1]) if abs(pitch_deg) > 70 else np.array([0,1,0])
    right = np.cross(up_ref, fwd)
    if np.linalg.norm(right) < 1e-8:
        up_ref = np.array([1,0,0]); right = np.cross(up_ref, fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right); up /= np.linalg.norm(up)
    return right, up, fwd

def frustum_edges_erp_centered(yaw_deg, pitch_deg, fov_deg, n_edge_samples=160):
    f = math.tan(math.radians(fov_deg) / 2.0)
    t = np.linspace(-1.0, 1.0, n_edge_samples)
    R, U, F = look_at_basis(yaw_deg, pitch_deg)
    edges = []
    for A, B in [(+U, +R), (+R, +U), (-U, +R), (-R, +U)]:
        rays = (F[None, :] + f * A[None, :] + (f * t)[:, None] * B[None, :])
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        u, v = sphere_to_erp(rays[:, 0], rays[:, 1], rays[:, 2])
        edges.append((u, v))
    return edges

# --------------------------- Plot helpers ------------------------------

def plot_wrapped_polyline(ax, u, v, *, color, lw=1.5, zorder=3):
    u = np.asarray(u); v = np.asarray(v)
    jumps = np.where(np.abs(np.diff(u)) > 0.5)[0]
    start = 0
    for j in list(jumps) + [len(u) - 1]:
        ax.plot(u[start:j+1], v[start:j+1], color=color, linewidth=lw, zorder=zorder)
        start = j + 1

def set_degree_axes(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect(0.5)
    xt = np.linspace(0, 1, 9); ax.set_xticks(xt)
    ax.set_xticklabels([f"{int(d)}°" for d in np.linspace(0, 360, 9)])
    yt = np.linspace(0, 1, 7); ax.set_yticks(yt)
    ax.set_yticklabels([f"{int(d)}°" for d in np.linspace(90, -90, 7)])
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")

def center_uv(yaw_deg, pitch_deg):
    u = (yaw_deg % 360.0) / 360.0
    v = (90.0 - pitch_deg) / 180.0
    return u, v

# ---------------------- Color palette per ring (robust) ----------------

def prepare_ring_palettes():
    """
    각 pitch 링마다:
      - yaws: np.linspace(...)로 생성한 yaw 리스트
      - colors: 해당 링의 고정 색 배열 (len = count)
    둘을 dict[pitch_deg] = (yaws, colors)로 저장.
    이후 어떤 루프 순서로 그리더라도 (pitch,yaw)로 색을 안정 조회.
    """
    rings = {
        90.0: 4, 77.5: 8, 45.0: 11, 22.5: 14,
        0.0: 15, -22.5: 14, -45.0: 11, -77.5: 8, -90.0: 4
    }
    palettes = {}
    for pitch_deg, count in rings.items():
        yaws = np.linspace(0.0, 360.0, num=count, endpoint=False)
        cmap = cm.get_cmap("tab20")
        colors = cmap(np.linspace(0.0, 1.0, count, endpoint=False))  # (count,4)
        palettes[pitch_deg] = (yaws, colors)
    return palettes

def wrap_angle_diff(a, b):
    """경도 차이 a-b 를 [-180,180)로 래핑한 절댓값(도)."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)

def color_for(palettes, pitch_deg, yaw_deg):
    """
    (pitch,yaw)로부터 색을 조회.
    정수 인덱스 의존 대신, 링의 yaw 리스트에서 가장 가까운 yaw를 찾아 매칭.
    (루프 순서가 달라도 항상 같은 색)
    """
    yaws, colors = palettes[pitch_deg]
    # 가장 가까운 yaw 인덱스 찾기 (래핑 고려)
    diffs = np.array([wrap_angle_diff(yaw_deg, y) for y in yaws])
    idx = int(diffs.argmin())
    return colors[idx]

# ---------------------------- DRAW ONE RING ----------------------------

def draw_one_ring(ax, pitch_deg, count, fov_deg, *, palettes, n_edge_samples=160):
    """
    한 링(pitch)을 그리되, 색은 palettes에서 (pitch,yaw)로 조회하여
    '센터 X'와 '4 edges'가 항상 동일 색을 사용.
    """
    yaws = np.linspace(0.0, 360.0, num=count, endpoint=False)
    for yaw_deg in yaws:
        c = color_for(palettes, pitch_deg, yaw_deg)
        cu, cv = center_uv(yaw_deg, pitch_deg)
        ax.scatter([cu], [cv], s=35, marker='x', color=c, linewidths=2, zorder=4)
        edges = frustum_edges_erp_centered(yaw_deg, pitch_deg, fov_deg, n_edge_samples=n_edge_samples)
        for eu, ev in edges:
            plot_wrapped_polyline(ax, eu, ev, color=c, lw=1.5, zorder=3)

# ---------------------------- Renderers --------------------------------

def render_combined(N_points=3000, fov_deg=80.0, fig_size=(12, 6),
                    out_path=None, show_scatter=True,
                    title="ERP + 89-view footprints (FOV 80°) — color-matched via (pitch,yaw)"):
    x, y, z = fibonacci_sphere(N_points)
    u, v = sphere_to_erp(x, y, z)
    views = make_89_views()

    palettes = prepare_ring_palettes()  # ★ 고정 팔레트 준비

    fig = plt.figure(figsize=fig_size); ax = fig.add_subplot(111)
    if show_scatter:
        ax.scatter(u, v, s=3, alpha=0.35, color="#7aa6c2", zorder=1)

    # views 순서와 무관하게, 항상 (pitch,yaw)로 색을 뽑아 사용
    for yaw_deg, pitch_deg in [(vw[0], vw[1]) for vw in views]:
        c = color_for(palettes, pitch_deg, yaw_deg)
        cu, cv = center_uv(yaw_deg, pitch_deg)
        ax.scatter([cu], [cv], s=25, marker='x', color=c, linewidths=2, zorder=4)
        edges = frustum_edges_erp_centered(yaw_deg, pitch_deg, fov_deg, n_edge_samples=128)
        for eu, ev in edges:
            plot_wrapped_polyline(ax, eu, ev, color=c, lw=1.0, zorder=3)

    set_degree_axes(ax)
    ax.set_title(title)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    return fig, ax

def render_per_ring(fov_deg=80.0, fig_size=(12, 6), out_dir="./out_rings",
                    N_points=3000, show_scatter=True):
    os.makedirs(out_dir, exist_ok=True)
    x, y, z = fibonacci_sphere(N_points)
    u_bg, v_bg = sphere_to_erp(x, y, z)

    palettes = prepare_ring_palettes()  # ★ 고정 팔레트 준비

    rings = {
        90.0: 4, 77.5: 8, 45.0: 11, 22.5: 14,
        0.0: 15, -22.5: 14, -45.0: 11, -77.5: 8, -90.0: 4
    }
    saved = []
    for pitch_deg, count in rings.items():
        fig = plt.figure(figsize=fig_size); ax = fig.add_subplot(111)
        if show_scatter:
            ax.scatter(u_bg, v_bg, s=2, alpha=0.25, color="#7aa6c2", zorder=1)
        draw_one_ring(ax, pitch_deg, count, fov_deg,
                      palettes=palettes, n_edge_samples=160)
        set_degree_axes(ax)
        ax.set_title(f"Ring {pitch_deg}° (FOV {fov_deg}°) — color-matched via (pitch,yaw)")
        fname = f"erp_ring_{str(pitch_deg).replace('.', '_')}deg.png"
        path = os.path.join(out_dir, fname)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        saved.append(path)
    return saved

# ------------------------------ Example --------------------------------
if __name__ == "__main__":
    # 1) 전체 합본
    render_combined(
        N_points=3000, fov_deg=80.0,
        out_path="./out/erp_89views_colormatched.png",
        show_scatter=True
    )
    # 2) 링별 저장
    paths = render_per_ring(
        fov_deg=80.0, out_dir="./out/rings",
        N_points=3000, show_scatter=True
    )
    print("Saved:", paths)
