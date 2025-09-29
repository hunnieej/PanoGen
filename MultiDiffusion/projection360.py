import torch
import math
import csv
import torch.nn.functional as F

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
def make_extrinsic(phi_deg: float, theta_deg: float, roll_deg: float = 0.0,
                               device=None, dtype=torch.float32):
    """
    Y-forward, Z-up 카메라 기준.
    월드→카메라 회전행렬 R을 오일러(Yaw=φ around Z)→(Pitch=θ around X)→(Roll=ρ around Y) 순서로 구성.
    반환은 camera-basis (x_right, y_forward, z_up)를 column으로 쌓은 R_cw.
    """
    device = device or torch.device('cpu')
    phi   = math.radians(phi_deg)    # yaw (around Z)
    theta = math.radians(theta_deg)  # pitch (around X)
    rho   = math.radians(roll_deg)   # roll (around Y)

    # 회전행렬: R = Rz(phi) @ Rx(theta) @ Ry(rho)
    cz, sz = math.cos(phi),   math.sin(phi)
    cx, sx = math.cos(theta), math.sin(theta)
    cy, sy = math.cos(rho),   math.sin(rho)

    Rz = torch.tensor([[ cz, -sz, 0.],
                       [ sz,  cz, 0.],
                       [0.,  0.,  1.]], device=device, dtype=dtype)
    Rx = torch.tensor([[1.,  0.,  0.],
                       [0.,  cx, -sx],
                       [0.,  sx,  cx]], device=device, dtype=dtype)
    Ry = torch.tensor([[ cy, 0.,  sy],
                       [0.,  1.,  0.],
                       [-sy, 0.,  cy]], device=device, dtype=dtype)

    R = Rz @ Rx @ Ry

    # 열벡터가 camera basis: x(right), y(forward), z(up)
    x = torch.nn.functional.normalize(R[:,0], dim=0)
    y = torch.nn.functional.normalize(R[:,1], dim=0)
    z = torch.linalg.cross(x, y)
    R_cw = torch.stack([x, y, z], dim=-1)
    return R_cw


def make_view_centers_89(): # SphereDiff 논문 세팅
    phi = []
    theta = []
    
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

    phi = [v[0] for v in views] #longitudes
    theta = [v[1] for v in views] # latitudes

    return phi, theta

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

# --------- Dynamic Latent Sampling ---------
@torch.no_grad()
def dynamic_sampling(uv: torch.Tensor, K: torch.Tensor, Hp: int, Wp: int):
    """
    Center-first(링 우선)는 유지하되, 빈칸 없이 타일을 채우도록
    '샘플(uv)'와 '격자(rc)'를 동일 규칙(링→방위각)으로 전역 정렬하고
    앞에서부터 1:1로 매칭한다. (스파이럴 격자, 센터우선 유지)
    """
    device = uv.device
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # camera-normalized
    u_n = (uv[:,0] - cx) / fx
    v_n = (uv[:,1] - cy) / fy

    # unit-square [-1,1] 정규화
    tan_hfov = (Wp / (2.0 * fx)).item()
    u_hat = u_n / tan_hfov
    v_hat = v_n / tan_hfov

    # uv의 링/방위각
    r  = torch.sqrt(torch.clamp(u_hat**2 + v_hat**2, min=0.0))
    az = torch.atan2(v_hat, u_hat) % (2.0 * math.pi)
    mid = Hp // 2
    ring_uv = torch.clamp((torch.clamp(r, 0.0, 1.0) * mid).long(), 0, mid - 1)

    # 스파이럴 격자 전체
    rc = spiral_coords_even(Hp, Wp).to(device)      # [H*W,2]
    # 격자셀 중심 기준 방위각
    cxg, cyg = mid - 0.5, mid - 0.5
    dy = (rc[:,0].float() + 0.5) - cyg
    dx = (rc[:,1].float() + 0.5) - cxg
    az_rc = torch.atan2(dy, dx) % (2.0 * math.pi)

    # 격자 링 인덱스: 스파이럴 정의 그대로
    # (거리 대신 정수 링: 바깥으로 갈수록 링 번호 증가)
    # 아래는 rc가 어느 링에 속하는지 직접 계산
    # 중심 2x2가 ring=0, 그 바깥 사각 둘레가 ring=1, ...
    ring_rc = torch.empty(rc.shape[0], dtype=torch.long, device=device)
    m = mid
    for i in range(m):
        top, left  = m - i - 1, m - i - 1
        bot, right = m + i,     m + i
        # 사각 링 경계에 해당하는 좌표만 해당 링
        on_top = (rc[:,0] == top) & (rc[:,1] >= left) & (rc[:,1] <= right)
        on_bot = (rc[:,0] == bot) & (rc[:,1] >= left) & (rc[:,1] <= right)
        on_lft = (rc[:,1] == left) & (rc[:,0] >  top) & (rc[:,0] <  bot)
        on_rgt = (rc[:,1] == right) & (rc[:,0] >  top) & (rc[:,0] <  bot)
        mask_i = on_top | on_bot | on_lft | on_rgt
        ring_rc[mask_i] = i

    # 전역 정렬 키 (링 → 방위각 → 안정화용 인덱스)
    M = uv.shape[0]
    base_uv = torch.arange(M, device=device).float()
    key_uv  = ring_uv.float() * 1e6 + (az / (2.0 * math.pi)) * 1e3 + base_uv / (M + 1.0)
    order_uv = torch.argsort(key_uv)

    Ntar = rc.shape[0]
    base_rc = torch.arange(Ntar, device=device).float()
    key_rc  = ring_rc.float() * 1e6 + (az_rc / (2.0 * math.pi)) * 1e3 + base_rc / (Ntar + 1.0)
    order_rc = torch.argsort(key_rc)

    # 선택 개수
    Ksel = min(M, Ntar)
    pick   = order_uv[:Ksel]                      # uv에서 뽑을 순서
    rc_sel = rc[order_rc[:Ksel]]                  # 대응하는 스파이럴 격자 칸
    lin    = rc_sel[:,0] * Wp + rc_sel[:,1]       # linear index

    return Hp, Wp, pick, lin, rc_sel

# ------------------------------------------------------------------------
@torch.no_grad()
def project_dynamic_sampling(dirs: torch.Tensor, R: torch.Tensor, K: torch.Tensor,
                             Hp: int, Wp: int, fov_deg: float = 80.0):
    dirs_f = dirs.to(torch.float32)
    R_f    = R.to(torch.float32)
    K_f    = K.to(torch.float32)
    X = dirs_f @ R_f.T  # [N,3]
    fx, fy, cx, cy = K_f[0,0], K_f[1,1], K_f[0,2], K_f[1,2]
    
    h = torch.atan(Wp / (2.0 * fx))

    X = dirs_f @ R_f.T
    Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)  # 단위벡터
    cos_half_fov = math.cos(0.5*math.radians(fov_deg))
    in_view = (Xn[:,1] >= cos_half_fov)

    vis_idx = torch.nonzero(in_view, as_tuple=False).squeeze(1)
    if vis_idx.numel() == 0: #view 2개 문제 있음 (0,90) / (0,-90)
        print("[OK][project_dynamic_sampling Y-forward] empty view after FoV filter, M=0")
        uv = X.new_zeros((0, 2))
        Hp_out, Wp_out, pick_in_vis, lin_coords_tok, rc_coords_tok = dynamic_sampling(uv, K_f, Hp, Wp)
        used_idx = vis_idx  # empty
        return Hp_out, Wp_out, used_idx, lin_coords_tok, rc_coords_tok, uv

    Xv = X[vis_idx]
    yv = Xv[:,1].clamp_min(1e-6)
    xy = Xv[:,0] / yv
    zy = Xv[:,2] / yv
    u  = fx * xy + cx
    v  = fy * zy + cy
    uv = torch.stack([u, v], dim=-1)

    tan_hfov = torch.tan(h).item()
    if uv.numel() > 0:
        u_n = (uv[:,0] - cx) / fx
        v_n = (uv[:,1] - cy) / fy
        u_hat = u_n / tan_hfov
        v_hat = v_n / tan_hfov
        print("[OK][project_dynamic_sampling Y-forward]",
              f"u:[{u_hat.min().item():.3f},{u_hat.max().item():.3f}] ",
              f"v:[{v_hat.min().item():.3f},{v_hat.max().item():.3f}] ",
              f"M={uv.shape[0]}")
    else:
        print("[OK][project_dynamic_sampling Y-forward] empty view after projection, M=0")

    Hp_out, Wp_out, pick_in_vis, lin_coords_tok, rc_coords_tok = dynamic_sampling(uv, K_f, Hp, Wp)
    used_idx = vis_idx[pick_in_vis]
    return Hp_out, Wp_out, used_idx, lin_coords_tok, rc_coords_tok, uv[pick_in_vis]

# --------- 89-view 타일 생성 ----------
def spherical_to_perspective_tiles(
    dirs: torch.Tensor,
    Hp: int = 32, Wp: int = 32,
    fov_deg: float = 80.0,
    *,
    phi_theta_provider = None,
    device=None, dtype=None
):
    device = device or dirs.device
    dtype  = dtype  or dirs.dtype
    K  = make_intrinsic(Hp, Wp, fov_deg, device=device, dtype=dtype)

    if phi_theta_provider is None:
        phi, theta = make_view_centers_89()
    else:
        phi, theta = phi_theta_provider()

    tiles = []
    for phi_deg, theta_deg in zip(phi, theta):
        phi = math.radians(phi_deg); theta = math.radians(theta_deg)

        # (+Y forward = Z-up)
        look_dir = torch.tensor([
            math.cos(theta)*math.cos(phi),
            math.cos(theta)*math.sin(phi),
            math.sin(theta)
        ], device=device, dtype=dtype)

        R = make_extrinsic(phi_deg=float(phi_deg),
                               theta_deg=float(theta_deg),
                               roll_deg=0.0,
                               device=device, dtype=dtype)

        Hp_out, Wp_out, used_idx, lin_tok, rc_tok, uv_sel = project_dynamic_sampling(
            dirs, R, K, Hp, Wp, fov_deg
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
            "phi": phi_deg, "theta": theta_deg,
            "R": R, "K": K,
            "H": Hp_out, "W": Wp_out,
            "used_idx": used_idx,
            "lin_coords": lin_tok,
            "rc_coords": rc_tok,
            "uv_unit": uv_unit,
            "center_dir": look_dir,
        })
    return tiles

##########################################################################################################################
# ------------------------------------------------------------------------
# F^-1 : I -> S
# ------------------------------------------------------------------------
@torch.no_grad()
def fuse_perspective_to_spherical(tile_feats, tiles_info, sphere_dirs, *, tau=0.5):
    device = tile_feats.device
    N_dirs, C = sphere_dirs.shape[0], tile_feats.shape[1]

    accum = torch.zeros(N_dirs, C, device=device, dtype=torch.float32)
    wsum  = torch.zeros(N_dirs, 1, device=device, dtype=torch.float32)
    sdirs = sphere_dirs / (sphere_dirs.norm(dim=-1, keepdim=True) + 1e-8)

    tau = float(tau)
    if not (tau > 0):
        tau = 1e-6

    for tile_idx, tile in enumerate(tiles_info):
        feat_hw = tile_feats[tile_idx]                          # [C,Ht,Wt]
        uid = tile['used_idx'].long()                           # [M]
        lin = tile['lin_coords'].long()                         # [M]
        if uid.numel() == 0:
            continue
        assert lin.numel() == uid.numel(), "lin/uid length mismatch"

        perm = torch.argsort(lin)
        lin  = lin[perm]; uid = uid[perm]
        feat_flat = feat_hw.reshape(C, -1).permute(1, 0).contiguous().float()  # [Ht*Wt,C]
        feat_sel  = feat_flat.index_select(0, lin)                              # [M,C])
        d = sdirs.index_select(0, uid)                                         # [M,3]
        c = tile['center_dir'].to(device).float()
        c = c / (c.norm() + 1e-8)
        dot = (d * c.unsqueeze(0)).sum(-1).clamp(-1.0, 1.0)                    # [M]
        r   = torch.arccos(dot)                                                # [M]
        w   = torch.exp(-r / tau).unsqueeze(1).to(torch.float32)               # [M,1]
        accum.index_add_(0, uid, w * feat_sel)                                  # [N,C]
        wsum.index_add_(0,  uid, w)                                             # [N,1]

    return accum, wsum

##########################################################################################################################
# ------------------------------------------------------------------------
# Logging Functions for Debugging
# ------------------------------------------------------------------------
@torch.no_grad()
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
