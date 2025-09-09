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
    KAIST Response
    - view-direction에 대한 정보는 diffusion model에 입력되지 않고, Patch간 정보 교환은 overlapping region에서 이뤄짐
    - Initial latent space = spherical latent
    - 사용하는 spherical latents의 개수(N=2,600)가 resolution에 영향을 주고, 최종 resize된 size는 1024*2048
    - SANA의 NoPE와 관계없이 view direction을 바꿔가며, 32x32 latent sampling 하고 일반 2D diffusion model로 작동
'''
########################################################################################################
# NOTE : N개의 spherical latent 방향 벡터를 균등하게 분포
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

def make_view_centers_89(): # SphereDiff 논문 세팅
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
# NOTE : Dynamic Latent Sampling for DiT
# Paper Comment : 1) a queue, 2) FoV adjustment, and 3) center-first selection
# For DiT-based Models (SD 3.5, SANA)
# ------------------------------------------------------------------------

def _ring_coords_even(H: int, i: int):
    mid = H // 2
    top, left  = mid - i,     mid - i
    bot, right = mid + i - 1, mid + i - 1
    coords = []
    for c in range(left, right + 1):      coords.append((top, c))
    for r in range(top + 1, bot):         coords.append((r, right))
    for c in range(right, left - 1, -1):  coords.append((bot, c))
    for r in range(bot - 1, top, -1):     coords.append((r, left))
    return coords

def _spiral_coords_even(H: int, W: int):
    assert H == W and H % 2 == 0, "Hp,Wp must be even and equal."
    order = []
    for i in range(1, H // 2 + 1):
        order.extend(_ring_coords_even(H, i))
    return torch.tensor(order, dtype=torch.long)

# --------- Dynamic Latent Sampling: 토큰 격자 타겟 ----------
@torch.no_grad()
def dynamic_sampling_to_tokens(uv: torch.Tensor, K: torch.Tensor, Hp: int, Wp: int):
    device, dtype = uv.device, uv.dtype
    fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]
    u_n = (uv[:,0] - cx) / fx
    v_n = (uv[:,1] - cy) / fy
    r2  = u_n**2 + v_n**2
    order = torch.argsort(r2)  # center-first

    rc = _spiral_coords_even(Hp, Wp).to(device)               # [Hp*Wp,2]
    Ntar = rc.shape[0]
    Ksel = min(uv.shape[0], Ntar)
    pick = order[:Ksel]                                       # 중심부터 선택
    rc_sel = rc[:Ksel]
    lin = rc_sel[:,0] * Wp + rc_sel[:,1]
    return Hp, Wp, pick, lin, rc_sel

@torch.no_grad()
def project_dynamic_sampling_token(dirs: torch.Tensor, R: torch.Tensor, K: torch.Tensor,
                                   Hp: int, Wp: int):
    X = dirs @ R.T            # [N,3] cam 좌표
    vis = X[:,2] > 1e-6       # 정면만 사용
    vis_idx = torch.nonzero(vis, as_tuple=False).squeeze(1)
    Xv = X[vis]

    homog = (K @ Xv.T).T
    uv = torch.stack([homog[:,0]/homog[:,2], homog[:,1]/homog[:,2]], dim=-1)  # [M,2]

    Hp, Wp, pick_in_vis, lin_coords_tok, rc_coords_tok = dynamic_sampling_to_tokens(uv, K, Hp, Wp)
    used_idx = vis_idx[pick_in_vis]

    return Hp, Wp, used_idx, lin_coords_tok, rc_coords_tok, uv[pick_in_vis]

# --------- 89-view 타일 생성(토큰 타겟) ----------
def spherical_to_perspective_tiles_tokenized(
    dirs: torch.Tensor,
    Hp: int = 32, Wp: int = 32,          # ← 모델별 토큰 타겟 (SANA/SD3.5: 32x32 등)
    fov_deg: float = 80.0,
    *,
    yaw_pitch_provider = None,            # 기본 None이면 make_view_centers_89 사용
    device=None, dtype=None
):
    device = device or dirs.device
    dtype  = dtype  or dirs.dtype
    K  = make_intrinsic(Hp, Wp, fov_deg, device=device, dtype=dtype)

    # 89개 뷰(center) 가져오기
    if yaw_pitch_provider is None:
        yaws, pitches = make_view_centers_89()
    else:
        yaws, pitches = yaw_pitch_provider()

    tiles = []
    for yaw_deg, pitch_deg in zip(yaws, pitches):
        yaw = math.radians(yaw_deg); pitch = math.radians(pitch_deg)

        # (+Z forward)
        look_dir = torch.tensor([
            math.cos(pitch)*math.sin(yaw),
            math.sin(pitch),
            math.cos(pitch)*math.cos(yaw)
        ], device=device, dtype=dtype)

        R = make_extrinsic(look_dir)

        Hp_out, Wp_out, used_idx, lin_tok, rc_tok, uv_sel = project_dynamic_sampling_token(
            dirs, R, K, Hp, Wp
        )

        tiles.append({
            "yaw": yaw_deg, "pitch": pitch_deg,
            "R": R, "K": K,
            "H": Hp_out, "W": Wp_out,
            "used_idx": used_idx,
            "lin_coords": lin_tok,
            "rc_coords": rc_tok,
            "uv_sel": uv_sel,
        })
    return tiles

##########################################################################################################################
# ------------------------------------------------------------------------
# F^-1 : I -> S
# ----------------------------------------------------------------------
@torch.no_grad()
def fuse_perspective_to_spherical(
    tile_feats: torch.Tensor,   # [N_tiles, C, Ht, Wt]  (픽셀 그리드)  또는 [N_tiles, C, Hp, Wp] (토큰 그리드)
    tiles_info: list,           # 각 dict: 'used_idx' [M], 'lin_coords' [M], ('uv_sel' 옵션), 'K' 또는 ('Hp','Wp')
    sphere_dirs: torch.Tensor,  # [N_dirs,3]  (인터페이스 유지용, 여기선 사용 X)
    *,
    tau: float = 0.5,           # center-prior 스케일 (정규화 평면 거리 r에 대한 exp(-r/tau))
    w_scheme: str = "uv"        # "uv" | "grid"
):
    device = tile_feats.device
    N_dirs = sphere_dirs.shape[0]
    C = tile_feats.shape[1]

    # 누적은 fp32로
    accum = torch.zeros(N_dirs, C, device=device, dtype=torch.float32)
    wsum  = torch.zeros(N_dirs, 1, device=device, dtype=torch.float32)

    for tile_idx, tile in enumerate(tiles_info):
        Ht, Wt = int(tile['H']), int(tile['W'])
        feat_hw = tile_feats[tile_idx]                           # [C,Ht,Wt]
        feat_flat = feat_hw.reshape(C, -1).permute(1, 0)         # [Ht*Wt, C]

        lin = tile['lin_coords'].long()                          # [M]
        used_idx = tile['used_idx'].long()                       # [M]

        # 선택된 위치의 피처만 뽑기
        feat_sel = feat_flat.index_select(0, lin).to(torch.float32)     # [M,C]

        # ---- 가중치 계산
        if w_scheme == "uv" and ('uv_sel' in tile) and (tile['uv_sel'] is not None):
            uv = tile['uv_sel']
            K  = tile['K'].to(uv.dtype)
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            u_n = (uv[:, 0] - cx) / fx
            v_n = (uv[:, 1] - cy) / fy
            r   = torch.sqrt(u_n**2 + v_n**2)                    # [M]
        else:
            rr = lin // Wt
            cc = lin %  Wt
            cy = (Ht - 1) * 0.5
            cx = (Wt - 1) * 0.5
            dx = (cc.to(torch.float32) - cx) / max(Wt * 0.5, 1.0)
            dy = (rr.to(torch.float32) - cy) / max(Ht * 0.5, 1.0)
            r  = torch.sqrt(dx*dx + dy*dy)                       # [M]

        w = torch.exp(-r / max(tau, 1e-6)).to(torch.float32).unsqueeze(1)  # [M,1]

        accum.index_add_(0, used_idx, w * feat_sel)              # [N_dirs,C]
        wsum.index_add_(0,  used_idx, w)                         # [N_dirs,1]

    wsum = torch.where(wsum > 0, wsum, torch.ones_like(wsum))
    spherical_feats = (accum / wsum).to(tile_feats.dtype)        # [N_dirs,C]
    return spherical_feats


##########################################################################################################################
# NOTE : ERP Splatting for visualization
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
    tile_img01: torch.Tensor,   # [3,Himg,Wimg] in [0,1]
    R_wc: torch.Tensor,         # [3,3] world<-camera  (== R.T)
    K_px: torch.Tensor,         # [3,3] intrinsics (pixel)
    pano_H: int, pano_W: int,
    fov_deg: float, beta: float = 1.0, use_coslat=True,
    accum_img: torch.Tensor = None, wsum: torch.Tensor = None
):
    device = tile_img01.device
    _, Himg, Wimg = tile_img01.shape

    if accum_img is None:
        accum_img = torch.zeros(3, pano_H, pano_W, device=device, dtype=torch.float32)
    if wsum is None:
        wsum = torch.zeros(1, pano_H, pano_W, device=device, dtype=torch.float32)

    jj = torch.arange(Himg, device=device, dtype=torch.float32) + 0.5
    ii = torch.arange(Wimg, device=device, dtype=torch.float32) + 0.5
    v, u = torch.meshgrid(jj, ii, indexing='ij')

    fx, fy, cx, cy = [K_px[0,0], K_px[1,1], K_px[0,2], K_px[1,2]]
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = torch.ones_like(x)
    d_cam = torch.stack([x, y, z], dim=-1)
    d_cam = torch.nn.functional.normalize(d_cam, dim=-1)

    R_wc = R_wc.to(torch.float32)
    d_world = torch.einsum('hwj,jk->hwk', d_cam, R_wc.T)

    lon = torch.atan2(d_world[..., 0], d_world[..., 2])
    lat = torch.asin(d_world[..., 1].clamp(-1, 1))
    U = (lon / (2*math.pi) + 0.5) * pano_W
    V = (0.5 - lat / math.pi) * pano_H

    t = math.tan(math.radians(fov_deg * 0.5))
    r_edge = torch.maximum(torch.abs(x)/t, torch.abs(y)/t).clamp(0, 1)
    w_edge = torch.cos(0.5*math.pi*r_edge) ** 2
    w_geo  = (d_cam[..., 2].clamp_min(0)) ** beta
    w_erp  = torch.cos(lat).abs() if use_coslat else 1.0
    w = (w_edge * w_geo * w_erp).to(torch.float32).unsqueeze(0)

    u0 = torch.floor(U).long() % pano_W
    u1 = (u0 + 1) % pano_W
    v0 = torch.clamp(torch.floor(V), 0, pano_H-2).long()
    du = (U - u0.float()); dv = (V - v0.float())
    w00 = (1-du)*(1-dv); w10 = du*(1-dv); w01 = (1-du)*dv; w11 = du*dv

    for (wu, offy, use_u1) in [(w00, 0, False), (w10, 0, True), (w01, 1, False), (w11, 1, True)]:
        w_full = w * wu.unsqueeze(0)
        yy = v0 if offy == 0 else torch.clamp(v0 + 1, 0, pano_H - 1)
        xx = u0 if not use_u1 else u1
        idx = (yy * pano_W + xx).reshape(-1)

        for c in range(3):
            accum_img[c].view(-1).index_add_(0, idx, (tile_img01[c].to(torch.float32) * w_full.squeeze(0)).reshape(-1))
        wsum.view(-1).index_add_(0, idx, w_full.squeeze(0).reshape(-1))

    return accum_img, wsum

@torch.no_grad()
def stitch_final_erp_from_tiles_SANA(
    feats_spherical: torch.Tensor,   # [N_dirs, C]
    tiles: list,                     # dict: 'H','W','K','R','lin_coords','used_idx'
    vae : float,
    pano_H: int, pano_W: int, fov_deg: float,
):
    device = feats_spherical.device
    s = get_vae_spatial_factor(vae)

    vae.eval()
    vae.requires_grad_(False)
    vae = vae.to(dtype=torch.float32, device=device)
    vae_scale = float(getattr(vae.config, "scaling_factor", 0.41407))

    accum = torch.zeros(3, pano_H, pano_W, device=device, dtype=torch.float32)
    wsum  = torch.zeros(1, pano_H, pano_W, device=device, dtype=torch.float32)

    for tile in tiles:
        Ht, Wt = tile['H'], tile['W']
        K_lat  = tile['K']; K_px = scale_intrinsics_to_pixels(K_lat, s).to(torch.float32)
        R_wc   = tile['R'].to(torch.float32)

        flat = torch.zeros(Ht*Wt, feats_spherical.shape[1], device=device, dtype=feats_spherical.dtype)
        flat.index_copy_(0, tile['lin_coords'].long(), feats_spherical[tile['used_idx'].long()])
        L_chw = flat.view(Ht, Wt, -1).permute(2,0,1).contiguous()  # [C,Ht,Wt]
        tile_latent = L_chw.unsqueeze(0)
        tile_latent = tile_latent.to(torch.bfloat16) / vae_scale
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                img_tile = vae.decode(tile_latent).sample[0].to(torch.float32)

        img01 = (img_tile / 2 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]

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

    sf = float(getattr(vae.config, "scaling_factor", 1.5305))
    sh = float(getattr(vae.config, "shift_factor",   0.0609))

    force_upcast = bool(getattr(vae.config, "force_upcast", False))
    if force_upcast:
        vae = vae.to(dtype=torch.float32)
        vae_in_dtype = torch.float32
        use_autocast = False
    else:
        vae_in_dtype = getattr(vae, "dtype", torch.bfloat16)
        use_autocast = (vae_in_dtype in (torch.bfloat16, torch.float16))

    accum = torch.zeros(3, pano_H, pano_W, device=device, dtype=torch.float32)
    wsum  = torch.zeros(1, pano_H, pano_W, device=device, dtype=torch.float32)

    for tile in tiles:
        Ht, Wt = tile['H'], tile['W']
        K_lat  = tile['K']
        K_px   = scale_intrinsics_to_pixels(K_lat, s).to(torch.float32)
        R_wc   = tile['R'].to(torch.float32)

        flat = torch.zeros(Ht*Wt, feats_spherical.shape[1], device=device, dtype=feats_spherical.dtype)
        flat.index_copy_(0, tile['lin_coords'].long(), feats_spherical[tile['used_idx'].long()])
        L_chw = flat.view(Ht, Wt, -1).permute(2,0,1).contiguous()  # [C,Ht,Wt]

        tile_latent = L_chw.unsqueeze(0).to(torch.float32)         # [1,C,Ht,Wt], calc in fp32
        tile_latent = (tile_latent / sf) + sh
        tile_latent = tile_latent.to(device=vae.device, dtype=vae_in_dtype)

        if use_autocast and vae_in_dtype != torch.float32:
            with torch.cuda.amp.autocast(enabled=True, dtype=vae_in_dtype):
                img_tile = vae.decode(tile_latent).sample[0].to(torch.float32)  # [-1,1] -> fp32로 모음
        else:
            img_tile = vae.decode(tile_latent).sample[0].to(torch.float32)

        img01 = (img_tile / 2 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]

        accum, wsum = splat_tile_to_erp(
            img01, R_wc, K_px, pano_H, pano_W, fov_deg,
            beta=2.0, use_coslat=True,
            accum_img=accum, wsum=wsum
        )

    wsum[wsum == 0] = 1.0
    erp = (accum / wsum).clamp(0, 1)  # [3,H,W]
    return erp
