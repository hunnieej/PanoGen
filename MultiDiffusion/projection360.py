import torch
import math
import csv

########################################################################################################
# NOTE : Sphere - S^2 - ERP구현
'''
    - Differential Renderer 사용해서 최적화 (교수님 피드백)
    - View 수 조절 가능하게 parameterization (Paper: 89) : Future work로 더 작은 view 언급
    - P = K[R|t] : Projection matrix, t = [0,0,0] (Camera at origin)
    - tilde u = P * tilde d = (u', v', w').T
    - u = (u'/w', v'/w').T : normalized pixel coordinates
    KAIST Response
    - view-direction에 대한 정보는 diffusion model에 입력되지 않고, Patch간 정보 교환은 overlapping region에서 이뤄짐
    - Initial latent space = spherical latent
    - 사용하는 spherical latents의 개수(N=2,600)가 resolution에 영향을 주고, 최종 resize된 size는 1024*2048
    - SANA의 NoPE와 관계없이 view direction을 바꿔가며, 32x32 latent sampling 하고 일반 2D diffusion model로 작동
'''
########################################################################################################
# NOTE : N개의 spherical latent 방향 벡터를 균등하게 분포    
def generate_fibonacci_lattice(N):
    # https://arxiv.org/abs/1607.04590
    # https://observablehq.com/@meetamit/fibonacci-lattices
    # https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    phi = (1 + math.sqrt(5.0)) / 2.0
    indices = torch.arange(N, dtype=torch.float32) + 0.5
    z = 1.0 - 2.0 * indices / N                                     # in (-1,1)
    theta = 2.0 * math.pi * torch.remainder(indices / phi, 1.0)     # {i φ^{-1}}
    rho = torch.sqrt(torch.clamp(1.0 - z**2, min=0.0))
    x = rho * torch.cos(theta); y = rho * torch.sin(theta)
    coords = torch.stack([x, y, z], 1)
    return coords

# NOTE : Intrinsic matrix K Generation
def make_intrinsic(H, W, fov_deg, device=None, dtype=torch.float32):
    device = device or torch.device('cpu')
    fov = math.radians(fov_deg)
    fx = W / (2.0 * math.tan(fov / 2.0))
    fy = fx
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0., cx],
                      [0., fy, cy],
                      [0., 0., 1.]], device=device, dtype=dtype)
    return K

# NOTE : Extrinsic matrix R Generation (R_cw)
# World 좌표계에서 Camera 좌표계로 변환 : X_cam = R.T @ X_world
def make_extrinsic(look_dir: torch.Tensor, yaw_deg: float = None, eps: float = 1e-6):
    z = torch.nn.functional.normalize(look_dir, dim=0)
    if yaw_deg is not None:
        yaw = math.radians(yaw_deg)
        up_hint = torch.tensor([math.sin(yaw), 0.0, math.cos(yaw)], device=z.device, dtype=z.dtype)
    else:
        up_hint = torch.tensor([0., 1., 0.], device=z.device, dtype=z.dtype)

    if torch.abs(torch.dot(z, up_hint)) > 1.0 - eps:
        tmp = torch.tensor([1., 0., 0.], device=z.device, dtype=z.dtype)
        if torch.abs(torch.dot(z, tmp)) > 1.0 - eps:
            tmp = torch.tensor([0., 1., 0.], device=z.device, dtype=z.dtype)
        up_hint = torch.nn.functional.normalize(torch.linalg.cross(z, tmp), dim=0)

    x = torch.nn.functional.normalize(torch.linalg.cross(up_hint, z), dim=0)
    y = torch.linalg.cross(z, x)
    return torch.stack([x, y, z], dim=-1)
    
def make_view_centers_89(): # SphereDiff 논문 세팅
    yaws = []
    pitches = []
    
    def ring(phi_deg, num_theta):
        return [(round((360.0/num_theta)*k, 6), phi_deg) for k in range(num_theta)]
    
    views = []
    views += ring(0.0, 15)
    views += ring(+22.5, 14)
    views += ring(-22.5, 14)
    views += ring(+45.0, 11)
    views += ring(-45.0, 11)
    views += ring(+77.5, 8)
    views += ring(-77.5, 8)
    views += ring(+90.0, 4)
    views += ring(-90.0, 4)

    yaws = [v[0] for v in views]
    pitches = [v[1] for v in views]

    return yaws, pitches

####################################################################################################################
# ------------------------------------------------------------------------
# NOTE : Dynamic Latent Sampling for DiT
# Paper Comment : 1) a queue, 2) FoV adjustment, and 3) center-first selection
# Paper Comments 중 FoV adjustment 구현하지 않음
# For DiT-based Models (SD 3.5, SANA)
# ------------------------------------------------------------------------
@torch.no_grad()
def spiral_coords_even(H: int, W: int, *, return_linear: bool = False, device=None):
    """
    짝수 정사각형 격자(H==W, H%2==0)에 대해
    중심(2x2 블록)에서 바깥으로 확장되는 '스파이럴 순서'의 좌표를 반환.

    Args:
        H, W (int): 격자 높이/너비. 반드시 H==W 이고 짝수여야 함.
        return_linear (bool): True면 (row,col) 대신 선형 인덱스(row*W+col) 반환.
        device: 반환 텐서의 device.

    Returns:
        torch.LongTensor:
          - return_linear=False: shape [H*W, 2], 각 원소가 (row, col)
          - return_linear=True : shape [H*W],    각 원소가 linear index (row*W+col)
    """
    assert H == W and (H % 2 == 0), "Hp,Wp must be even and equal."
    device = device or torch.device('cpu')

    mid = H // 2
    out_rc = []

    for i in range(1, mid + 1):
        # N x N rectangle Generation
        top, left  = mid - i,     mid - i
        bot, right = mid + i - 1, mid + i - 1
        for c in range(left, right + 1):
            out_rc.append((top, c))
        for r in range(top + 1, bot):
            out_rc.append((r, right))
        for c in range(right, left - 1, -1):
            out_rc.append((bot, c))
        for r in range(bot - 1, top, -1):
            out_rc.append((r, left))

    rc = torch.tensor(out_rc, dtype=torch.long, device=device)

    if return_linear:
        lin = rc[:, 0] * W + rc[:, 1]
        return lin
    return rc

# --------- Dynamic Latent Sampling ----------
@torch.no_grad()
def dynamic_sampling(uv: torch.Tensor, K: torch.Tensor, Hp: int, Wp: int):
    device = uv.device
    fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]

    # camera-normalized
    u_n = (uv[:,0] - cx) / fx
    v_n = (uv[:,1] - cy) / fy

    # unit-square 좌표 ([-1,1]^2) — 경계가 정확히 ±1
    tan_hfov = (Wp / (2.0 * fx)).item()
    u_hat = u_n / tan_hfov
    v_hat = v_n / tan_hfov

    # 중심 우선(반지름^2) 정렬 — FOV/해상도 불변
    r2  = u_hat**2 + v_hat**2
    order = torch.argsort(r2)

    rc = spiral_coords_even(Hp, Wp).to(device)
    Ntar = rc.shape[0]
    Ksel = min(uv.shape[0], Ntar)
    pick = order[:Ksel]
    rc_sel = rc[:Ksel]
    lin = rc_sel[:,0] * Wp + rc_sel[:,1]
    return Hp, Wp, pick, lin, rc_sel

# ------------------------------------------------------------------------
@torch.no_grad()
def project_dynamic_sampling(dirs: torch.Tensor, R: torch.Tensor, K: torch.Tensor,
                                   Hp: int, Wp: int, z_eps: float = 1e-3):
    dirs_f = dirs.to(torch.float32)
    R_f    = R.to(torch.float32)
    K_f    = K.to(torch.float32)

    # Camera 좌표계: X_cam = X_world @ R.T
    X = dirs_f @ R_f.T
    vis = X[:,2] > z_eps
    vis_idx = torch.nonzero(vis, as_tuple=False).squeeze(1)
    Xv = X[vis]

    fx, fy, cx, cy = K_f[0,0], K_f[1,1], K_f[0,2], K_f[1,2]
    xz = Xv[:,0] / Xv[:,2]
    yz = Xv[:,1] / Xv[:,2]
    u  = fx * xz + cx
    v  = fy * yz + cy
    uv = torch.stack([u, v], dim=-1)

    tan_hfov = (Wp / (2.0 * fx)).item()
    r2 = xz**2 + yz**2
    in_fov = r2 <= (tan_hfov**2)

    if in_fov.any():
        Xv = Xv[in_fov]
        uv = uv[in_fov]
        vis_idx = vis_idx[in_fov]
    else:
        uv = uv[:0]
        vis_idx = vis_idx[:0]

    M = uv.shape[0]
    if M > 0:
        u_n = (uv[:,0] - cx) / fx
        v_n = (uv[:,1] - cy) / fy
        u_hat = u_n / tan_hfov
        v_hat = v_n / tan_hfov
        print("[OK][project_dynamic_sampling]",
            f"u:[{u_hat.min().item():.3f},{u_hat.max().item():.3f}] ",
            f"v:[{v_hat.min().item():.3f},{v_hat.max().item():.3f}] ",
            f"M={M}")
    else:
        print("[OK][project_dynamic_sampling] empty view after FoV filter, M=0")
    Hp, Wp, pick_in_vis, lin_coords_tok, rc_coords_tok = dynamic_sampling(uv, K_f, Hp, Wp)
    used_idx = vis_idx[pick_in_vis]
    return Hp, Wp, used_idx, lin_coords_tok, rc_coords_tok, uv[pick_in_vis]

# --------- 89-view 타일 생성 ----------
def spherical_to_perspective_tiles(
    dirs: torch.Tensor,
    Hp: int = 32, Wp: int = 32,
    fov_deg: float = 80.0,
    *,
    yaw_pitch_provider = None,
    device=None, dtype=None
):
    device = device or dirs.device
    dtype  = dtype  or dirs.dtype
    K  = make_intrinsic(Hp, Wp, fov_deg, device=device, dtype=dtype)

    if yaw_pitch_provider is None:
        yaws, pitches = make_view_centers_89()
    else:
        yaws, pitches = yaw_pitch_provider()

    tiles = []
    for yaw_deg, pitch_deg in zip(yaws, pitches):
        yaw = math.radians(yaw_deg); pitch = math.radians(pitch_deg)

        # (+Z forward = Y-up)
        # look_dir = torch.tensor([
        #     math.cos(pitch)*math.sin(yaw),
        #     math.sin(pitch),
        #     math.cos(pitch)*math.cos(yaw)
        # ], device=device, dtype=dtype)

        # (+Y forward = Z-up)
        look_dir = torch.tensor([
            math.cos(pitch)*math.cos(yaw),
            math.cos(pitch)*math.sin(yaw),
            math.sin(pitch)
        ], device=device, dtype=dtype)

        R = make_extrinsic(look_dir)

        Hp_out, Wp_out, used_idx, lin_tok, rc_tok, uv_sel = project_dynamic_sampling(
            dirs, R, K, Hp, Wp
        )
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        if uv_sel.numel() > 0:
            u_n = (uv_sel[:,0] - cx) / fx
            v_n = (uv_sel[:,1] - cy) / fy
            tan_hfov = (Wp_out / (2.0 * fx)).item()
            uv_unit = torch.stack([u_n / tan_hfov, v_n / tan_hfov], dim=-1)
        else:
            uv_unit = uv_sel

        tiles.append({
            "yaw": yaw_deg, "pitch": pitch_deg,
            "R": R, "K": K,
            "H": Hp_out, "W": Wp_out,
            "used_idx": used_idx,
            "lin_coords": lin_tok,
            "rc_coords": rc_tok,
            "uv_unit": uv_unit,
        })
    return tiles
# ------------------------------------------------------------------------
def log_overlap_between_views(tiles, log_path="overlap_log.txt"):
    num_views = len(tiles)
    overlaps = []
    
    with open(log_path, "w") as f:
        for i in range(num_views):
            set_i = set(tiles[i]['used_idx'].cpu().tolist())
            size_i = len(set_i)
            for j in range(i+1, num_views):
                set_j = set(tiles[j]['used_idx'].cpu().tolist())
                intersection = set_i.intersection(set_j)
                overlap_count = len(intersection)
                overlap_ratio_i = overlap_count / size_i if size_i > 0 else 0.0
                size_j = len(set_j)
                overlap_ratio_j = overlap_count / size_j if size_j > 0 else 0.0
                
                log_line = (f"View {i} and View {j}: Overlap count = {overlap_count}, "
                            f"Overlap ratio (view {i}) = {overlap_ratio_i:.3f}, "
                            f"Overlap ratio (view {j}) = {overlap_ratio_j:.3f}\n")
                
                f.write(log_line)
                overlaps.append({
                    'view_i': i, 'view_j': j,
                    'overlap_count': overlap_count,
                    'overlap_ratio_i': overlap_ratio_i,
                    'overlap_ratio_j': overlap_ratio_j,
                })
    print(f"Overlap log saved at {log_path}")
    return overlaps

@torch.no_grad()
def compute_and_save_overlap_matrix(tiles, log_path="overlap_matrix.csv", 
                                    mode="count"):
    n = len(tiles)
    matrix = torch.zeros((n, n), dtype=torch.float32)
    
    indices_list = [set(tile['used_idx'].cpu().tolist()) for tile in tiles]
    sizes = [len(s) for s in indices_list]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0 if mode != "count" else sizes[i]
            else:
                intersection_count = len(indices_list[i].intersection(indices_list[j]))
                if mode == "count":
                    matrix[i, j] = intersection_count
                elif mode == "ratio_i":
                    matrix[i, j] = intersection_count / sizes[i] if sizes[i] > 0 else 0.0
                elif mode == "ratio_j":
                    matrix[i, j] = intersection_count / sizes[j] if sizes[j] > 0 else 0.0

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([""] + [f"View_{i}" for i in range(n)])
        for i in range(n):
            row = [f"View_{i}"] + matrix[i].tolist()
            writer.writerow(row)
    
    print(f"Overlap matrix saved to {log_path}")
    return matrix

##########################################################################################################################
# ------------------------------------------------------------------------
# F^-1 : I -> S
# ----------------------------------------------------------------------
@torch.no_grad()
def fuse_perspective_to_spherical(
    tile_feats: torch.Tensor,    # [N_tiles, C, Ht, Wt]
    tiles_info: list,            # 각 dict: 'used_idx','lin_coords','uv_sel','K','H','W'
    sphere_dirs: torch.Tensor,   # [N_dirs, 3] (실제로는 index 크기만 사용)
    *,
    tau: float = 0.5,
):
    device = tile_feats.device
    N_dirs = sphere_dirs.shape[0]
    C = tile_feats.shape[1]

    accum = torch.zeros(N_dirs, C, device=device, dtype=torch.float32)
    wsum  = torch.zeros(N_dirs, 1, device=device, dtype=torch.float32)

    for tile_idx, tile in enumerate(tiles_info):
        feat_hw = tile_feats[tile_idx]                            # [C,Ht,Wt]
        feat_flat = feat_hw.reshape(C, -1).permute(1, 0)          # [Ht*Wt, C]

        lin = tile['lin_coords'].long()                           # [M]
        used_idx = tile['used_idx'].long()                        # [M]
        feat_sel = feat_flat.index_select(0, lin).to(torch.float32)  # [M,C]

        uv = tile['uv_unit']
        u_hat = uv[:, 0]
        v_hat = uv[:, 1]
        r = torch.sqrt(u_hat**2 + v_hat**2)   # [0, sqrt(2)]
        w = torch.exp(-r /tau).unsqueeze(1).to(torch.float32)

        # ---- accumulate ----
        accum.index_add_(0, used_idx, w * feat_sel)
        wsum.index_add_(0,  used_idx, w)

    return accum, wsum

@torch.no_grad()
def fuse_persp_to_sph_with_raw(
    tile_feats: torch.Tensor,   # [1,C,Ht,Wt] (단일 view의 타일 묶음이면 리스트로 루프 돌리기)
    tile: dict,                 # 단일 타일(dict): 'used_idx','lin_coords','uv_sel','K','H','W'
    sphere_dirs: torch.Tensor,
    tau: float = 0.5
):
    # --- 기존 weighted ---
    acc_w, wsum = fuse_perspective_to_spherical(
        tile_feats=tile_feats, tiles_info=[tile], sphere_dirs=sphere_dirs, tau=tau
    )  # acc_w:[N_dirs,C], wsum:[N_dirs,1]

    # --- raw(비가중) 합 / 개수 (같은 used_idx로 원시 feature 평균을 보기 위함) ---
    C = tile_feats.shape[1]
    Ht, Wt = int(tile['H']), int(tile['W'])
    feat_hw = tile_feats.squeeze(0)                  # [C,Ht,Wt]
    feat_flat = feat_hw.reshape(C, -1).permute(1,0)  # [Ht*Wt, C]

    lin      = tile['lin_coords'].long()             # [M]
    used_idx = tile['used_idx'].long()               # [M]
    feat_sel = feat_flat.index_select(0, lin).to(torch.float32)  # [M,C]

    N_dirs = acc_w.shape[0]
    acc_raw = torch.zeros(N_dirs, C, device=feat_sel.device, dtype=torch.float32)
    cnt_raw = torch.zeros(N_dirs, 1, device=feat_sel.device, dtype=torch.float32)

    ones = torch.ones(used_idx.shape[0], 1, device=feat_sel.device, dtype=torch.float32)
    acc_raw.index_add_(0, used_idx, feat_sel)  # 비가중 합
    cnt_raw.index_add_(0, used_idx, ones)      # 개수

    return acc_w, wsum, acc_raw, cnt_raw

def build_membership_map(tiles, N_dirs):
    # idx -> list of views that include this index
    members = [[] for _ in range(N_dirs)]
    for v, t in enumerate(tiles):
        if t['used_idx'] is None or len(t['used_idx']) == 0: 
            continue
        for k in t['used_idx'].tolist():
            members[k].append(v)
    return members  # length N_dirs, each a python list of view ids

##########################################################################################################################
