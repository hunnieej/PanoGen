import torch
import math
from utils import get_vae_spatial_factor

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
    z = -(v - cy) / fy
    y = torch.ones_like(x)
    d_cam = torch.stack([x, y, z], dim=-1)
    d_cam = torch.nn.functional.normalize(d_cam, dim=-1)

    R_wc = R_wc.to(torch.float32)
    d_world = torch.einsum('hwj,jk->hwk', d_cam, R_wc.T)

    lon = torch.atan2(d_world[..., 0], d_world[..., 1])
    lat = torch.asin(d_world[..., 2].clamp(-1, 1))
    U = (lon / (2*math.pi) + 0.5) * pano_W
    V = (0.5 - lat / math.pi) * pano_H

    t = math.tan(math.radians(fov_deg * 0.5))
    r_edge = torch.maximum(torch.abs(x)/t, torch.abs(z)/t).clamp(0, 1)
    w_edge = torch.cos(0.5*math.pi*r_edge) ** 2
    w_geo  = (d_cam[..., 1].clamp_min(0)) ** beta
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
def stitch_final_erp_from_tiles(
    feats_spherical: torch.Tensor,   # [N_dirs, C]
    tiles: list,                     # dict: 'H','W','K','R','lin_coords','used_idx'
    vae,                             # AutoencoderKL
    pano_H: int, pano_W: int, fov_deg: float,
    rf_version: str = "3.5",         # "3.5" | "SANA"
):
    device = feats_spherical.device
    s = get_vae_spatial_factor(vae)

    vae.eval().requires_grad_(False).to(device=device)

    if rf_version == "SANA":
        vae = vae.to(dtype=torch.float32, device=device)
        vae_scale = float(getattr(vae.config, "scaling_factor", 0.41407))
        sf, sh = vae_scale, 0.0
        force_upcast = False
        vae_in_dtype = torch.bfloat16
        use_autocast = True

    elif rf_version == "3.5":
        sf = float(getattr(vae.config, "scaling_factor", 1.5305))
        sh = float(getattr(vae.config, "shift_factor", 0.0609))
        force_upcast = bool(getattr(vae.config, "force_upcast", False))

        if force_upcast:
            vae = vae.to(dtype=torch.float32)
            vae_in_dtype = torch.float32
            use_autocast = False
        else:
            vae_in_dtype = getattr(vae, "dtype", torch.bfloat16)
            use_autocast = (vae_in_dtype in (torch.bfloat16, torch.float16))
    else:
        raise ValueError(f"Unknown rf_version: {rf_version}")

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

        if rf_version.lower() == "SANA":
            tile_latent = L_chw.unsqueeze(0).to(torch.bfloat16) / sf
        else:  # 3.5
            tile_latent = L_chw.unsqueeze(0).to(torch.float32)
            tile_latent = (tile_latent / sf) + sh
            tile_latent = tile_latent.to(device=vae.device, dtype=vae_in_dtype)

        if use_autocast and vae_in_dtype != torch.float32:
            with torch.cuda.amp.autocast(enabled=True, dtype=vae_in_dtype):
                img_tile = vae.decode(tile_latent).sample[0].to(torch.float32)
        else:
            img_tile = vae.decode(tile_latent).sample[0].to(torch.float32)

        img01 = (img_tile / 2 + 0.5).clamp(0, 1)

        # splat to ERP
        accum, wsum = splat_tile_to_erp(
            img01, R_wc, K_px, pano_H, pano_W, fov_deg,
            beta=2.0, use_coslat=True,
            accum_img=accum, wsum=wsum
        )

    wsum[wsum == 0] = 1.0
    erp = (accum / wsum).clamp(0, 1)  # [3,H,W]
    return erp
