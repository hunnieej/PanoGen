import os, math
import numpy as np
import torch
import matplotlib.pyplot as plt

# --------------------------
# 기본 유틸
# --------------------------
def fibonacci_sphere(n_points: int, device="cpu", dtype=torch.float32):
    i = torch.arange(n_points, dtype=dtype, device=device)
    y = 1.0 - 2.0 * (i + 0.5) / n_points
    r = torch.sqrt(torch.clamp(1.0 - y * y, 0.0, 1.0))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = golden_angle * i
    x = r * torch.cos(theta)
    z = r * torch.sin(theta)
    dirs = torch.stack([x, y, z], dim=-1)  # [N,3], +X right, +Y up, +Z forward
    return torch.nn.functional.normalize(dirs, dim=-1)

def make_89_views():
    def ring(pitch_deg, n):
        return [(float(yaw), float(pitch_deg)) for yaw in np.linspace(0.0, 360.0, num=n, endpoint=False)]
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

def make_extrinsic(look_dir: torch.Tensor, up_hint: torch.Tensor = None, eps: float = 1e-5):
    z = torch.nn.functional.normalize(look_dir, dim=0)
    if up_hint is None:
        up_hint = torch.tensor([0., 1., 0.], device=z.device, dtype=z.dtype)
    if torch.abs(torch.dot(z, up_hint)) > 1.0 - eps:
        tmp = torch.tensor([1., 0., 0.], device=z.device, dtype=z.dtype)
        if torch.abs(torch.dot(z, tmp)) > 1.0 - eps:
            tmp = torch.tensor([0., 1., 0.], device=z.device, dtype=z.dtype)
        up_hint = torch.nn.functional.normalize(torch.linalg.cross(z, tmp), dim=0)
    x = torch.nn.functional.normalize(torch.linalg.cross(up_hint, z), dim=0)
    y = torch.linalg.cross(z, x)
    # return world->cam
    return torch.stack([x, y, z], dim=-1)

def lonlat_from_dirs(d: torch.Tensor):
    # d: [..., 3]
    lon = torch.atan2(d[..., 2], d[..., 0])                 # [-pi, pi]
    lat = torch.asin(d[..., 1].clamp(-1, 1))                # [-pi/2, pi/2]
    return lon, lat

# --------------------------
# View 투영 / 마스크
# --------------------------
def project_to_view_mask(dirs: torch.Tensor, yaw_deg: float, pitch_deg: float, fov_deg: float):
    """
    dirs: [N,3] world directions (unit)
    return:
      idx_vis: LongTensor[ M ] indices of visible dirs
      cam_xy:  Tensor[ M, 2 ] normalized camera plane coords x=(u-cx)/fx, y=(v-cy)/fy
      ang_xy:  Tensor[ M, 2 ] local angles (deg): (theta_x, theta_y) = (atan2(x,1), atan2(y,1))
    """
    device = dirs.device
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # (+Z forward) 시야 벡터
    look_dir = torch.tensor([math.cos(pitch)*math.sin(yaw),
                             math.sin(pitch),
                             math.cos(pitch)*math.cos(yaw)], device=device, dtype=dirs.dtype)
    R_wc = make_extrinsic(look_dir)         # world->cam
    X = dirs @ R_wc                         # [N,3], cam coords
    z = X[:, 2]
    vis = z > 0.0                           # front-hemisphere
    idx_vis = torch.nonzero(vis, as_tuple=False).squeeze(1)
    Xv = X[vis]

    # FoV 사각형 판단: |x| <= tan(FOV/2), |y| <= tan(FOV/2) (정방 렌즈 가정)
    t = math.tan(math.radians(fov_deg * 0.5))
    x = Xv[:, 0] / Xv[:, 2]
    y = Xv[:, 1] / Xv[:, 2]
    in_fov = (x.abs() <= t) & (y.abs() <= t)
    idx_vis = idx_vis[in_fov]
    x = x[in_fov]
    y = y[in_fov]

    ang_x = torch.rad2deg(torch.atan(x))   # atan(x/1)
    ang_y = torch.rad2deg(torch.atan(y))
    cam_xy = torch.stack([x, y], dim=-1)
    ang_xy = torch.stack([ang_x, ang_y], dim=-1)
    return idx_vis, cam_xy, ang_xy

# --------------------------
# ERP 스캐터 (축=deg, 제목에 포인트 수)
# --------------------------
def save_erp_scatter(dirs: torch.Tensor, idx_set: torch.Tensor, title_prefix: str, outpath: str,
                     point_size=2, alpha=0.9):
    # 모든 점의 lon/lat
    lon, lat = lonlat_from_dirs(dirs)  # radians
    lon_deg = torch.rad2deg(lon).cpu().numpy()
    lat_deg = torch.rad2deg(lat).cpu().numpy()

    # 표시할 인덱스만 마스크
    mask = np.zeros(dirs.shape[0], dtype=bool)
    mask[idx_set.cpu().numpy()] = True

    plt.figure(figsize=(12, 4))
    # 배경(연한 회색) + 선택점(진한 색)
    plt.scatter(lon_deg[~mask], lat_deg[~mask], s=point_size, c="#cccccc", alpha=0.4)
    plt.scatter(lon_deg[mask],  lat_deg[mask],  s=point_size, c="#1f77b4", alpha=alpha)

    plt.xlim([-180, 180]); plt.ylim([90, -90])  # 위가 +90°, 아래가 -90°
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title(f"{title_prefix} | points: {mask.sum()}")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# --------------------------
# View 로컬 정사각 패널 (축=deg, FoV 박스 포함)
# --------------------------
def save_view_panel(ang_xy: torch.Tensor, fov_deg: float, title_prefix: str, outpath: str,
                    s=5, alpha=0.9):
    # ang_xy: [M,2] in degrees (theta_x, theta_y)
    a = ang_xy.detach().cpu().numpy()
    fx = fov_deg * 0.5
    fy = fov_deg * 0.5

    plt.figure(figsize=(5,5))
    # FoV 박스
    plt.plot([-fx, +fx, +fx, -fx, -fx], [-fy, -fy, +fy, +fy, -fy], 'k--', lw=1, alpha=0.7)
    # 포인트
    if a.shape[0] > 0:
        plt.scatter(a[:,0], a[:,1], s=s, c="#d62728", alpha=alpha)
    plt.xlim([-fx, +fx]); plt.ylim([+fy, -fy])  # 위가 +deg(상) 되도록 inversion
    plt.xlabel("Local horizontal angle (deg)")
    plt.ylabel("Local vertical angle (deg)")
    plt.title(f"{title_prefix} | points: {a.shape[0]}")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# ====== Even-square + Spiral order ======
def _even_square_from_count(M: int, max_side: int = None):
    side = int(math.sqrt(max(M, 1)))
    if side % 2 == 1:
        side -= 1
    side = max(side, 2)
    if max_side is not None:
        side = min(side, max_side)
        if side % 2 == 1:
            side -= 1
        side = max(side, 2)
    return side, side

def _ring_coords(H: int, i: int):
    mid = H // 2
    top, left  = mid - i,     mid - i
    bot, right = mid + i - 1, mid + i - 1
    coords = []
    for c in range(left, right + 1):             coords.append((top, c))
    for r in range(top + 1, bot):                 coords.append((r, right))
    for c in range(right, left - 1, -1):          coords.append((bot, c))
    for r in range(bot - 1, top, -1):             coords.append((r, left))
    return coords

def _spiral_order(H: int, W: int):
    assert H == W and H % 2 == 0
    order = []
    for i in range(1, H // 2 + 1):
        order.extend(_ring_coords(H, i))
    return order

# ====== Dynamic sampling (Queue + FoV adjust + Center-first) ======
@torch.no_grad()
def dynamic_sampling_with_queue_and_fov(
    cam_xy_full: torch.Tensor,      # [M0,2]  (x,y) = ((u-cx)/fx,(v-cy)/fy) - front hemi & 기본 FoV 통과분
    idx_vis_full: torch.Tensor,     # [M0]    cam_xy_full과 1:1 대응하는 전역 index
    used_mask: torch.Tensor,        # [N]     이미 선택된 전역 점은 True (Queue)
    fov_deg_init: float,            # 초기 FoV(deg) - 여기에서 축소/확장
    max_side: int = 32,             # Hp=Wp 상한 (예: 32 → 1024 토큰)
    target_fill: float = 1.00,      # 목표 수 ≈ target_fill * (max_side*max_side)
    max_iter: int = 12,             # FoV 조절 반복 횟수
    tol_ratio: float = 0.10,        # 허용 오차(±10%)
):
    """
    반환:
      Hp, Wp, pick_idx (전역 index[K]), rc_coords[K,2], ang_sel[K,2], fov_eff_deg
    """
    device = cam_xy_full.device
    # 0) Queue: 아직 선택되지 않은 점만
    avail_mask = ~used_mask.index_select(0, idx_vis_full)
    if avail_mask.sum() == 0:
        return 0, 0, idx_vis_full.new_empty((0,), dtype=torch.long), \
               torch.empty(0,2,dtype=torch.long,device=device), \
               cam_xy_full.new_empty((0,2)), fov_deg_init

    xy_all = cam_xy_full[avail_mask]      # [M,2]
    idx_all = idx_vis_full[avail_mask]    # [M]
    M = xy_all.shape[0]

    # 1) 목표 토큰 수
    target_tokens = min(max_side * max_side, M)
    if target_tokens <= 0:
        return 0, 0, idx_vis_full.new_empty((0,), dtype=torch.long), \
               torch.empty(0,2,dtype=torch.long,device=device), \
               cam_xy_full.new_empty((0,2)), fov_deg_init

    # 2) FoV 조절: |x|,|y| <= tan(fov/2)*scale
    t0 = math.tan(math.radians(fov_deg_init * 0.5))
    s_lo, s_hi = 0.5, 1.5   # 탐색 범위 (필요시 조절 가능)
    s = 1.0

    def count_with_scale(s):
        t_eff = t0 * s
        msk = (xy_all[:,0].abs() <= t_eff) & (xy_all[:,1].abs() <= t_eff)
        return msk, int(msk.sum().item())

    # 초기
    msk, M_eff = count_with_scale(s)
    lo_goal = int(target_tokens * (1 - tol_ratio))
    hi_goal = int(target_tokens * (1 + tol_ratio))

    # 이진 탐색 비슷하게 조절
    it = 0
    while (M_eff > hi_goal or M_eff < lo_goal) and it < max_iter:
        if M_eff > hi_goal:
            s_hi = s
            s = 0.5 * (s_lo + s_hi)
        else:
            s_lo = s
            s = 0.5 * (s_lo + s_hi)
        msk, M_eff = count_with_scale(s)
        it += 1

    # 최종 FoV 마스크 적용
    xy_eff = xy_all[msk]
    idx_eff = idx_all[msk]
    M_eff = xy_eff.shape[0]

    if M_eff == 0:
        return 0, 0, idx_vis_full.new_empty((0,), dtype=torch.long), \
               torch.empty(0,2,dtype=torch.long,device=device), \
               cam_xy_full.new_empty((0,2)), fov_deg_init * s

    # 3) 중심우선 정렬
    r2 = xy_eff[:,0]**2 + xy_eff[:,1]**2
    order = torch.argsort(r2)
    xy_sorted  = xy_eff[order]
    idx_sorted = idx_eff[order]

    # 4) 짝수 정사각 그리드 + 스파이럴 채우기
    Hp, Wp = _even_square_from_count(M_eff, max_side=max_side)
    K = min(M_eff, Hp * Wp)
    coords_spiral = _spiral_order(Hp, Wp)
    rc_coords = torch.tensor(coords_spiral[:K], device=device, dtype=torch.long)
    pick_idx  = idx_sorted[:K]
    ang_sel   = torch.rad2deg(torch.atan(xy_sorted[:K]))  # 각도 변환(참고용): atan(x), atan(y)
    ang_sel   = torch.stack([ang_sel[:,0], ang_sel[:,1]], dim=-1)

    fov_eff_deg = 2.0 * math.degrees(math.atan(t0 * s))
    return Hp, Wp, pick_idx, rc_coords, ang_sel, fov_eff_deg

@torch.no_grad()
def visualize_views_and_save_indices_dynamic(
    n_points: int = 15000,
    fov_deg: float = 80.0,
    outdir: str = "out_views_dynamic",
    device: str = "cuda",
    dtype=torch.float32,
    max_side: int = 32,          # SANA=32, (모델 토큰 해상도에 맞춰 조절)
    target_fill: float = 1.00,   # 목표 = target_fill * (max_side^2)
):
    os.makedirs(outdir, exist_ok=True)

    # 1) Fibonacci lattice
    dirs = fibonacci_sphere(n_points, device=device, dtype=dtype)  # [N,3]
    views = make_89_views()

    # 2) 전역 Queue
    used_mask = torch.zeros(dirs.shape[0], dtype=torch.bool, device=device)
    use_queue = False
    local_used = used_mask if use_queue else torch.zeros_like(used_mask)

    # 3) 통계(선택 횟수)
    picked_cover = torch.zeros(dirs.shape[0], dtype=torch.int32, device=device)
    fov_cover    = torch.zeros(dirs.shape[0], dtype=torch.int32, device=device)

    for i, (yaw_deg, pitch_deg) in enumerate(views):
        # (A) 기본 FoV inliers
        idx_vis, cam_xy, ang_xy = project_to_view_mask(dirs, yaw_deg, pitch_deg, fov_deg)
        fov_cover[idx_vis] += 1

        # 기본 FoV panel (참고용)
        save_view_panel(
            ang_xy, fov_deg,
            title_prefix=f"[BASE] View {i:03d} yaw={yaw_deg:.1f}, pitch={pitch_deg:.1f}",
            outpath=os.path.join(outdir, f"panel_view_{i:03d}_base.png"),
            s=6, alpha=0.85
        )
        
        # (B) 동적 샘플링(큐 + FoV 조절 + 중심우선)
        Hp, Wp, pick_idx, rc_coords, ang_sel, fov_eff_deg = dynamic_sampling_with_queue_and_fov(
            cam_xy_full=cam_xy,
            idx_vis_full=idx_vis,
            used_mask=local_used,
            fov_deg_init=fov_deg,
            max_side=max_side,
            target_fill=target_fill
        )

        # queue 업데이트 + 통계
        if pick_idx.numel() > 0:
            used_mask[pick_idx] = True
            picked_cover[pick_idx] += 1

        # (C) 저장: 인덱스
        np.savetxt(os.path.join(outdir, f"indices_view_{i:03d}_fov.txt"),
                   idx_vis.detach().cpu().numpy().astype(np.int64), fmt="%d")
        np.savetxt(os.path.join(outdir, f"indices_view_{i:03d}_picked.txt"),
                   pick_idx.detach().cpu().numpy().astype(np.int64), fmt="%d")

        # (D) ERP (선택된 점만 강조)
        save_erp_scatter(
            dirs, pick_idx,
            title_prefix=f"[PICKED] View {i:03d} (yaw={yaw_deg:.1f}, pitch={pitch_deg:.1f})  "
                         f"Hp×Wp={Hp}×{Wp}, K={pick_idx.numel()}, FoV_eff≈{fov_eff_deg:.1f}°",
            outpath=os.path.join(outdir, f"erp_view_{i:03d}_picked.png"),
            point_size=3, alpha=0.95
        )

        # (E) 동적 샘플링 panel (선택된 토큰만, FoV 박스는 초기값/제목에 eff FoV 표시)
        save_view_panel(
            ang_sel, fov_deg,   # 박스는 초기 FoV로 그리되 제목에 eff FoV 표기
            title_prefix=f"[PICKED] View {i:03d} Hp×Wp={Hp}×{Wp}, K={ang_sel.shape[0]}, "
                         f"FoV_eff≈{fov_eff_deg:.1f}°",
            outpath=os.path.join(outdir, f"panel_view_{i:03d}_picked.png"),
            s=10, alpha=0.95
        )

    # ---- 커버리지 맵/히스토그램 저장 (선택 횟수 / FoV 포함 횟수) ----
    lon, lat = lonlat_from_dirs(dirs)
    lon_deg = torch.rad2deg(lon).cpu().numpy()
    lat_deg = torch.rad2deg(lat).cpu().numpy()

    def _save_cov(vals, tag):
        arr = vals.cpu().numpy()
        plt.figure(figsize=(12,4))
        sc = plt.scatter(lon_deg, lat_deg, c=arr, s=4, cmap="viridis", alpha=0.9)
        plt.xlim([-180, 180]); plt.ylim([90, -90])
        plt.xlabel("Longitude (deg)"); plt.ylabel("Latitude (deg)")
        plt.title(f"{tag} coverage map | N={dirs.shape[0]}")
        plt.colorbar(sc, label="#views")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{tag}_coverage_map.png"), dpi=220)
        plt.close()

        uniq, cnt = np.unique(arr, return_counts=True)
        with open(os.path.join(outdir, f"{tag}_coverage_hist.txt"), "w") as f:
            for u, c in zip(uniq, cnt):
                f.write(f"{int(u)}\t{int(c)}\n")

    _save_cov(fov_cover,   "fov_inclusion")
    _save_cov(picked_cover,"picked_dynamic")

    print(f"[DONE] Saved panels (base & picked), ERP highlights, indices, and coverage under: {outdir}")

visualize_views_and_save_indices_dynamic(
    n_points=2600,
    fov_deg=80.0,
    outdir="out_views_dynamic",
    device="cuda",
    max_side=32,          # 모델 토큰 해상도에 맞게
    target_fill=1.00      # 1.0이면 32×32 꽉 채우도록 FoV 조절 시도
)
