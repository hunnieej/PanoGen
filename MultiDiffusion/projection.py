import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from utils import get_vae_spatial_factor

########################################################################################################
# NOTE : Sphere - S^2 - ERP구현
'''
    - Differential Renderer 사용해서 최적화 (교수님 피드백)
    - View 수 조절 가능하게 parameterization (Paper: 89) : Future work로 더 작은 view 언급
    - P = K[R|t] : Projection matrix, t = [0,0,0] (Camera at origin)
    - tilde u = P * tilde d = (u', v', w').T
    - u = (u'/w', v'/w').T : normalized pixel coordinates
'''
########################################################################################################
# NOTE : N개의 spherical latent 방향 벡터를 균등하게 분포시키는 함수
def generate_fibonacci_lattice(N):
    # https://observablehq.com/@meetamit/fibonacci-lattices
    # https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    print(f'[INFO] Generating fibonacci lattice with {N} points')
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    indices = torch.arange(N, dtype=torch.float32)
    z = 1 - 2 * indices / (N - 1)  # Uniform distribution in z: [1, -1]
    theta = 2 * np.pi * indices / phi
    rho = torch.sqrt(1 - z**2)  # Radius at height z
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)

    coords = torch.stack([x, y, z], dim=1)  # Shape: (N, 3)
    coords = coords / torch.norm(coords, dim=1, keepdim=True)
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
def make_extrinsic(look_dir: torch.Tensor,
              up_hint: torch.Tensor = None, eps: float = 1e-3) -> torch.Tensor:
    z = torch.nn.functional.normalize(look_dir, dim=0) #forward vector
    if up_hint is None:
        up_hint = torch.tensor([0., 1., 0.], device=z.device, dtype=z.dtype)
    if torch.abs(torch.dot(z, up_hint)) > 1.0-eps:
        tmp = torch.tensor([1., 0., 0.], device=z.device, dtype=z.dtype)
        if torch.abs(torch.dot(z, tmp)) > 1 - eps:
            tmp = torch.tensor([0., 1., 0.], device=z.device, dtype=z.dtype)
        up_hint = torch.nn.functional.normalize(torch.linalg.cross(z, tmp), dim=0)
    # Gram-Schmidt process
    x = torch.nn.functional.normalize(torch.linalg.cross(up_hint, z), dim=0)
    y = torch.linalg.cross(z, x)
    return torch.stack([x, y, z], dim=-1)  # [3,3] world->cam

# NOTE : make_view_centers는 주어진 FOV와 오버랩을 기반으로 360도 파노라마 뷰의 중심 좌표를 생성
def make_view_centers(fov_deg=80.0, overlap=0.6): # FOV: 80°, overlap: 60% (SphereDiff)
        stride = fov_deg * (1.0 - overlap)          # e.g., 32° for 80°, 60% overlap
        yaws = []
        a = 0.0
        while a < 360.0 - 1e-6:
            yaws.append(a)
            a += stride
        if abs(360.0 - (yaws[-1] if yaws else 360.0)) > 1e-3:
            yaws.append(360.0) 

        pmin = -90.0 + fov_deg / 2.0
        pmax = +90.0 - fov_deg / 2.0
        pitches = []
        b = pmin
        while b <= pmax + 1e-6:
            pitches.append(b)
            b += stride
        return yaws, pitches

def make_view_centers_89():
    '''
    SphereDiff 논문에 맞춰서 hard coding
    '''
    yaws = []
    pitches = []
    
    def ring(phi_deg, num_theta):
        return [(round((360.0/num_theta)*k, 6), phi_deg) for k in range(num_theta)]
    
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

    yaws = [v[0] for v in views]
    pitches = [v[1] for v in views]

    return yaws, pitches

####################################################################################################################

# ------------------------------------------------------------------------
# NOTE : Dynamic Latent Sampling ver.1
# 1) a queue, 2) FoV adjustment, and 3) center-first selection
# ------------------------------------------------------------------------
def even_square_from_count(M: int, max_side: int = None):
    side = int(math.sqrt(M))
    if side % 2 == 1:
        side -= 1
    side = max(side, 2)
    if max_side is not None:
        side = min(side, max_side)
        if side % 2 == 1:
            side -= 1
    return side, side  # H, W

def ring_coords(H: int, i: int):
    mid = H // 2
    top, left  = mid - i,     mid - i
    bot, right = mid + i - 1, mid + i - 1
    coords = []
    for c in range(left, right + 1):
        coords.append((top, c))
    for r in range(top + 1, bot):
        coords.append((r, right))
    for c in range(right, left - 1, -1):
        coords.append((bot, c))
    for r in range(bot - 1, top, -1):
        coords.append((r, left))
    return coords

def spiral_coords(H: int, W: int):
    assert H == W and H % 2 == 0, "H,W must be even and equal for this spiral."
    order = []
    for i in range(1, H // 2 + 1):
        order.extend(ring_coords(H, i))
    return order

def dynamic_sampling_indices(
    uv: torch.Tensor,
    Hmax: int = None,
    Wmax: int = None,
    *,
    K: torch.Tensor,                 # ← 필수: 픽셀좌표 uv를 K로 정규화
):
    """
    uv: (M,2)  투영된 픽셀 좌표 (u,v)
    K : (3,3)  intrinsic matrix (fx, fy, cx, cy)
    반환: H, W, pick, lin_coords, rc
    """
    device, dtype = uv.device, uv.dtype
    M = uv.shape[0]
    assert M > 0, "uv must contain at least one point."
    assert K is not None, "K (intrinsic) must be provided."

    # --- 1) 중심-우선 정렬: 정규화 카메라 평면 기준 (u→(u-cx)/fx, v→(v-cy)/fy) ---
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    r2 = ((uv[:, 0] - cx) / fx) ** 2 + ((uv[:, 1] - cy) / fy) ** 2
    order = torch.argsort(r2)  # center-first

    # --- 2) 타일 크기: 짝수 정사각 (⌊√M⌋ 기반) + 상한 적용 ---
    Hcand, Wcand = even_square_from_count(M, None)
    if (Hmax is not None) and (Wmax is not None):
        s = max(2, min(Hcand, Wcand, Hmax, Wmax))
        if s % 2 == 1:
            s -= 1
        s = max(2, s)
        H = W = s
    else:
        H, W = Hcand, Wcand  # already even square

    # --- 3) 스파이럴 좌표(중앙→바깥 링) 및 매칭 ---
    coords = spiral_coords(H, W)           # length = H*W
    Ksel = min(len(coords), M)
    pick = order[:Ksel]                    # 안쪽 포인트부터 Ksel개
    rc = torch.tensor(coords[:Ksel], device=device, dtype=torch.long)
    lin_coords = rc[:, 0] * W + rc[:, 1]

    return H, W, pick, lin_coords, rc
# ------------------------------------------------------------------------
def project_dynamic_sampling(dirs: torch.Tensor, R: torch.Tensor, K: torch.Tensor,
                             H: int = None, W: int = None):
    X = dirs @ R.T
    z = X[:, 2]
    vis = z > 1e-6
    vis_idx = torch.nonzero(vis, as_tuple=False).squeeze(1)
    Xv = X[vis]

    homog = (K @ Xv.T).T
    uv = torch.stack([homog[:, 0] / homog[:, 2], homog[:, 1] / homog[:, 2]], dim=-1)  # [M,2]

    H, W, pick_in_vis, lin_coords, rc_coords = dynamic_sampling_indices(uv, H, W, K=K)
    used_idx = vis_idx[pick_in_vis]
    uv_sel = uv[pick_in_vis]

    return H, W, used_idx, lin_coords, rc_coords, uv_sel

# ------------------------------------------------------------------------
# F : S -> I
# ----------------------------------------------------------------------
def spherical_to_perspective_tiles(dirs: torch.Tensor, H, W, fov_deg=80.0, overlap=0.6):
    K = make_intrinsic(H, W, fov_deg, device=dirs.device, dtype=dirs.dtype)
    # yaws, pitches = make_view_centers(fov_deg, overlap)
    yaws, pitches = make_view_centers_89()

    tiles = []
    for yaw_deg, pitch_deg in zip(yaws, pitches):
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        # +Z convention으로 설정
        look_dir = torch.tensor([math.cos(pitch)*math.sin(yaw),
                            math.sin(pitch),
                            math.cos(pitch)*math.cos(yaw)], 
                            device=dirs.device, dtype=dirs.dtype)
        # +X convention (ver.1)
        # look_dir = torch.tensor([math.cos(pitch)*math.cos(yaw),
        #                     math.sin(pitch),
        #                     math.cos(pitch)*math.sin(yaw)], 
        #                     device=dirs.device, dtype=dirs.dtype)
        R = make_extrinsic(look_dir)
        Ht, Wt, used_idx, lin_coords, rc_coords, uv_sel = project_dynamic_sampling(
            dirs, R, K, H=H, W=W
        )
        tiles.append({
            "yaw": yaw_deg, "pitch": pitch_deg,
            "R": R, "K": K,
            "H": Ht, "W": Wt,
            "used_idx": used_idx,
            "lin_coords": lin_coords,
            "rc_coords": rc_coords,
            "uv_sel": uv_sel,
        })
    return tiles

##########################################################################################################################
# ------------------------------------------------------------------------
# F^-1 : I -> S
# ----------------------------------------------------------------------
def perspective_to_spherical_latent(
    perspective_feats: torch.Tensor,   # [N_tiles, C, Ht, Wt]
    tiles_info: list,                  # 각 dict: 'used_idx','uv_sel','lin_coords','K','H','W' (R는 불필요)
    sphere_dirs: torch.Tensor,         # [N_dirs, 3] (여기선 사용 X, 인터페이스 유지용)
    tau: float = 0.5
):
    device = perspective_feats.device
    N_dirs = sphere_dirs.shape[0]
    C = perspective_feats.shape[1]
    dtype = perspective_feats.dtype
    accum = torch.zeros(N_dirs, C, device=device, dtype=dtype)
    wsum  = torch.zeros(N_dirs, 1, device=device, dtype=dtype)

    for tile_idx, tile in enumerate(tiles_info):
        Ht, Wt = tile['H'], tile['W']
        feat_hw = perspective_feats[tile_idx]                # [C, Ht, Wt]
        feat_flat = feat_hw.reshape(C, -1).permute(1, 0)     # [Ht*Wt, C]

        lin = tile['lin_coords'].long()                      # [M]
        used_idx = tile['used_idx'].long()                   # [M]
        uv = tile['uv_sel'].to(dtype)                                  # [M, 2]
        K = tile['K'].to(dtype)                                  # [3, 3]

        # --- pick the same M points as uv/used_idx ---
        feat_sel = feat_flat.index_select(0, lin)            # [M, C]

        # --- distortion-aware weight (center-prior) ---
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        u_n = (uv[:, 0] - cx) / fx
        v_n = (uv[:, 1] - cy) / fy
        r = torch.sqrt(u_n**2 + v_n**2)                      # [M]
        w = torch.exp(-r / tau).to(dtype).unsqueeze(1)                 # [M,1]

        accum.index_add_(0, used_idx, w * feat_sel)
        wsum.index_add_(0,  used_idx, w)

    spherical_feats = torch.where(
        wsum > 0,
        accum / (wsum + 1e-6),
        torch.zeros_like(accum),
    )
    return spherical_feats  # [N_dirs, C]

##########################################################################################################################

##########################################################################################################################
# NOTE : ERP Splatting
##########################################################################################################################

def scale_intrinsics_to_pixels(K_lat: torch.Tensor, scale: int):
    """K를 latent 해상도에서 pixel 해상도로 스케일 (보통 scale=VAE factor=8)"""
    K_px = K_lat.clone()
    K_px[0,0] *= scale; K_px[1,1] *= scale  # fx, fy
    K_px[0,2] *= scale; K_px[1,2] *= scale  # cx, cy
    return K_px

@torch.no_grad()
# NOTE : camera 정의는 구의 중심! ERP 할 때 부호 주의
def splat_tile_to_erp(
    tile_img01: torch.Tensor,   # [3, Himg, Wimg]  in [0,1]
    R_wc: torch.Tensor,         # [3,3] world<-camera
    K_px: torch.Tensor,         # [3,3] intrinsics in PIXELS for this decoded tile
    pano_H: int, pano_W: int,
    fov_deg: float, beta: float = 1.0, use_coslat=True,
    accum_img: torch.Tensor = None, wsum: torch.Tensor = None
):
    """
    타일 픽셀마다 카메라 광선 -> 월드 방향 -> ERP 좌표로 뿌리기(쌍선형 분배).
    가중치: 중심 페이드 + foreshortening(z_cam^beta) + (옵션) cos(lat).
    """
    device = tile_img01.device
    C, Himg, Wimg = tile_img01.shape
    if accum_img is None:
        accum_img = torch.zeros(3, pano_H, pano_W, device=device, dtype=tile_img01.dtype)
    if wsum is None:
        wsum = torch.zeros(1, pano_H, pano_W, device=device, dtype=tile_img01.dtype)

    # 픽셀 그리드 (center)
    jj = torch.arange(Himg, device=device, dtype=tile_img01.dtype) + 0.5
    ii = torch.arange(Wimg, device=device, dtype=tile_img01.dtype) + 0.5
    v, u = torch.meshgrid(jj, ii, indexing='ij')  # [Himg,Wimg]
    
    K_px = K_px.to(tile_img01.dtype)
    R_wc = R_wc.to(tile_img01.dtype)

    fx, fy = K_px[0,0], K_px[1,1]
    cx, cy = K_px[0,2], K_px[1,2]
    # 카메라 좌표계로 정규화된 광선
    # NOTE : 위아래 맞추려고 부호 반전
    x = (u - cx) / fx
    y = -(v - cy) / fy
    z = torch.ones_like(x)
    d_cam = torch.stack([x, y, z], dim=-1)                         # [H,W,3]
    d_cam = torch.nn.functional.normalize(d_cam, dim=-1)           # 단위벡터

    # 월드 방향 (R_wc: camera->world).  네 make_extrinsic는 world->cam 이므로 R_wc = R.T
    d_world = torch.einsum('hwj,jk->hwk', d_cam, R_wc.T)             # [H,W,3]

    # ERP 좌표
    lon = torch.atan2(d_world[..., 0], d_world[..., 2])            # [-π,π]
    # lon = torch.atan2(d_world[..., 2], d_world[..., 0])            # [-π,π] (ver.1)
    lat = torch.asin(d_world[..., 1].clamp(-1, 1))                 # [-π/2,π/2]
    U = (lon / (2*math.pi) + 0.5) * pano_W
    V = (0.5 - lat / math.pi) * pano_H

    # 가중치
    t = math.tan(math.radians(fov_deg/2))
    r_edge = torch.maximum(torch.abs(x)/t, torch.abs(y)/t).clamp(0, 1)      # [0,1]
    w_edge = torch.cos(0.5*math.pi*r_edge) ** 2                              # 중심 우대
    w_geo  = (d_cam[..., 2].clamp_min(0)) ** beta                            # foreshortening
    w_erp  = torch.cos(lat).abs() if use_coslat else 1.0
    w = (w_edge * w_geo * w_erp).unsqueeze(0)                                # [1,H,W]

    # bilinear splat
    # u0 = torch.clamp(torch.floor(U), 0, pano_W-2).long() # ver.1
    u0 = torch.floor(U).long() % pano_W
    u1 = (u0 + 1) % pano_W
    v0 = torch.clamp(torch.floor(V), 0, pano_H-2).long()
    du = (U - u0.float()); dv = (V - v0.float())
    w00 = (1-du)*(1-dv); w10 = du*(1-dv); w01 = (1-du)*dv; w11 = du*dv       # [H,W]

    # 누적
    for (wu, offy, use_u1) in [
        (w00, 0, False),
        (w10, 0, True),   # ← u1 사용
        (w01, 1, False),
        (w11, 1, True),   # ← u1 사용
    ]:
        w_full = (w * wu.unsqueeze(0))  # [1,H,W]

        # 수직: 클램프 (v1 = clamp(v0+1))
        yy = (v0 if offy == 0 else torch.clamp(v0 + 1, 0, pano_H - 1))
        # 수평: 래핑 (u1은 미리 u1 = (u0 + 1) % pano_W 로 만들어둔 값)
        xx = (u0 if not use_u1 else u1)

        idx = (yy * pano_W + xx).reshape(-1)

        for c in range(3):
            accum_img[c].view(-1).index_add_(0, idx, (tile_img01[c] * w_full.squeeze(0)).reshape(-1))
        wsum.view(-1).index_add_(0, idx, w_full.squeeze(0).reshape(-1))

    return accum_img, wsum

@torch.no_grad()
def stitch_final_erp_from_tiles(
    feats_spherical: torch.Tensor,   # [N_dirs, C]
    tiles: list,                     # dict: 'H','W','K','R','lin_coords','used_idx'
    vae, scale: float,               # <- 외부 scale 무시하고 VAE의 scaling_factor 사용 권장
    pano_H: int, pano_W: int, fov_deg: float,
    rf_version: str
):
    device = feats_spherical.device
    s = get_vae_spatial_factor(vae)

    vae.eval()
    vae.requires_grad_(False)
    vae = vae.to(dtype=torch.float32, device=device)
    vae_scale = float(getattr(vae.config, "scaling_factor", 0.13025))
    # VAE 입력 파이프의 dtype
    if rf_version.startswith('SANA'):
        # SANA VAE는 fp16 파라미터
        vae_in_dtype = torch.bfloat16
    else:
        try:
            vae_in_dtype = next(vae.post_quant_conv.parameters()).dtype
        except StopIteration:
            vae_in_dtype = torch.float32

    accum = torch.zeros(3, pano_H, pano_W, device=device, dtype=torch.float32)
    wsum  = torch.zeros(1, pano_H, pano_W, device=device, dtype=torch.float32)

    for tile in tiles:
        Ht, Wt = tile['H'], tile['W']
        K_lat  = tile['K']; K_px = scale_intrinsics_to_pixels(K_lat, s).to(torch.float32)
        R_wc   = tile['R'].to(torch.float32)

        # ---- 타일 latent 만들기: [C,Ht,Wt] ----
        flat = torch.zeros(Ht*Wt, feats_spherical.shape[1], device=device, dtype=feats_spherical.dtype)
        flat.index_copy_(0, tile['lin_coords'].long(), feats_spherical[tile['used_idx'].long()])
        L_chw = flat.view(Ht, Wt, -1).permute(2,0,1).contiguous()  # [C,Ht,Wt]

        # ---- 디코드: VAE만 fp32 + AMP OFF ----
        tile_latent = L_chw.unsqueeze(0)
        if vae_in_dtype == torch.float32:
            tile_latent = tile_latent.to(torch.float32) / vae_scale
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    img_tile = vae.decode(tile_latent).sample[0].to(torch.float32)

        elif vae_in_dtype == torch.float16:
            tile_latent = tile_latent.to(torch.float16) / vae_scale
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    img_tile = vae.decode(tile_latent).sample[0].to(torch.float32)

        elif vae_in_dtype == torch.bfloat16:
            tile_latent = tile_latent.to(torch.bfloat16) / vae_scale
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    img_tile = vae.decode(tile_latent).sample[0].to(torch.float32)

        else:
            raise ValueError(f"Unsupported VAE dtype: {vae_in_dtype}")
        img01 = (img_tile / 2 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]

        # ---- ERP로 누적 ----
        accum, wsum = splat_tile_to_erp(
            img01, R_wc, K_px, pano_H, pano_W, fov_deg,
            beta=2.0, use_coslat=True,
            accum_img=accum, wsum=wsum
        )

    wsum[wsum == 0] = 1.0
    erp = (accum / wsum).clamp(0, 1)  # [3,H,W]
    return erp

@torch.no_grad()
def stitch_final_erp_from_tiles_35(
    feats_spherical: torch.Tensor,   # [N_dirs, C]
    tiles: list,                     # dict: 'H','W','K','R','lin_coords','used_idx'
    vae,                             # AutoencoderKL
    pano_H: int, pano_W: int, fov_deg: float,
):
    device = feats_spherical.device
    s = get_vae_spatial_factor(vae)

    vae.eval().requires_grad_(False).to(device=device)

    # --- SD3.5 VAE 스케일/시프트 ---
    sf = float(getattr(vae.config, "scaling_factor", 1.5305))
    sh = float(getattr(vae.config, "shift_factor",   0.0609))

    # --- dtype 정책 결정 ---
    force_upcast = bool(getattr(vae.config, "force_upcast", False))
    if force_upcast:
        vae = vae.to(dtype=torch.float32)
        vae_in_dtype = torch.float32
        use_autocast = False           # fp32 디코딩
    else:
        # VAE가 bf16로 로드됐다면 그대로 유지
        vae_in_dtype = getattr(vae, "dtype", torch.bfloat16)
        # bf16/half면 autocast 사용 가능, fp32면 끔
        use_autocast = (vae_in_dtype in (torch.bfloat16, torch.float16))

    accum = torch.zeros(3, pano_H, pano_W, device=device, dtype=torch.float32)
    wsum  = torch.zeros(1, pano_H, pano_W, device=device, dtype=torch.float32)

    for tile in tiles:
        Ht, Wt = tile['H'], tile['W']
        K_lat  = tile['K']
        K_px   = scale_intrinsics_to_pixels(K_lat, s).to(torch.float32)
        R_wc   = tile['R'].to(torch.float32)

        # ---- 타일 latent 만들기: [C,Ht,Wt] ----
        flat = torch.zeros(Ht*Wt, feats_spherical.shape[1], device=device, dtype=feats_spherical.dtype)
        flat.index_copy_(0, tile['lin_coords'].long(), feats_spherical[tile['used_idx'].long()])
        L_chw = flat.view(Ht, Wt, -1).permute(2,0,1).contiguous()  # [C,Ht,Wt]

        # ---- 디코드 입력 준비: 항상 fp32에서 scale/shift 계산 -> 최종 dtype으로 캐스팅 ----
        tile_latent = L_chw.unsqueeze(0).to(torch.float32)         # [1,C,Ht,Wt], calc in fp32
        tile_latent = (tile_latent / sf) + sh
        tile_latent = tile_latent.to(device=vae.device, dtype=vae_in_dtype)

        # ---- 디코드: fp32면 autocast OFF, bf16/half면 해당 dtype으로 ON ----
        if use_autocast and vae_in_dtype != torch.float32:
            with torch.cuda.amp.autocast(enabled=True, dtype=vae_in_dtype):
                img_tile = vae.decode(tile_latent).sample[0].to(torch.float32)  # [-1,1] -> fp32로 모음
        else:
            # fp32 경로(권장 for SD3.5 force_upcast)
            img_tile = vae.decode(tile_latent).sample[0].to(torch.float32)

        img01 = (img_tile / 2 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]

        # ---- ERP 누적 ----
        accum, wsum = splat_tile_to_erp(
            img01, R_wc, K_px, pano_H, pano_W, fov_deg,
            beta=2.0, use_coslat=True,
            accum_img=accum, wsum=wsum
        )

    wsum[wsum == 0] = 1.0
    erp = (accum / wsum).clamp(0, 1)  # [3,H,W]
    return erp


@torch.no_grad()
def stitch_final_erp_from_tiles_xl(
    feats_spherical: torch.Tensor,   # [N_dirs, 4]
    dirs: torch.Tensor,              # [N_dirs, 3]
    tiles: list,                     # dict: 'H','W','K','R','lin_coords','used_idx'
    vae,
    pano_H: int, pano_W: int, fov_deg: float
):
    device = feats_spherical.device
    s = get_vae_spatial_factor(vae)  # latent→pixel intrinsics scale
    vae_scale = float(getattr(vae.config, "scaling_factor", 0.13025))

    # VAE는 fp32 + autocast OFF로 고정
    vae.eval()
    vae.requires_grad_(False)
    vae = vae.to(dtype=torch.float32)

    accum = torch.zeros(3, pano_H, pano_W, device=device, dtype=torch.float32)
    wsum  = torch.zeros(1, pano_H, pano_W, device=device, dtype=torch.float32)

    for tile in tiles:
        Ht, Wt = tile['H'], tile['W']
        K_px   = scale_intrinsics_to_pixels(tile['K'], s).to(torch.float32)
        R_wc   = tile['R'].to(torch.float32)

        # [4,Ht,Wt] latent (fp32)
        flat = torch.zeros(Ht*Wt, feats_spherical.shape[1], device=device, dtype=torch.float32)
        flat.index_copy_(0, tile['lin_coords'].long(), feats_spherical[tile['used_idx'].long()])
        L_chw = flat.view(Ht, Wt, -1).permute(2,0,1).contiguous()

        # 디코드: 반드시 latents / vae_scale (한 번만)
        latents = (L_chw.unsqueeze(0).to(torch.float32)) / vae_scale

        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
            dec = vae.decode(latents).sample[0].to(torch.float32)  # [-1,1]

        img01 = (dec / 2 + 0.5).clamp(0, 1)  # [3,Ht,Wt]

        accum, wsum = splat_tile_to_erp(
            img01, R_wc, K_px, pano_H, pano_W, fov_deg,
            beta=4.0, use_coslat=True,
            accum_img=accum, wsum=wsum
        )

    wsum[wsum == 0] = 1.0
    return (accum / wsum).clamp(0, 1)  # [3,H,W]