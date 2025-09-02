import torch
import torch.nn.functional as F
import numpy as np
import math
import os
from PIL import Image, ImageDraw

def build_2d_sincos_pe(Hp, Wp, D, device, dtype, temperature=10000.0):
    assert D % 2 == 0, "pos dim D must be even"
    d_half = D // 2
    assert d_half % 2 == 0, "D/2 must be even"

    def get_1d_sincos(L, dim):
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)     # [L,1]
        i = torch.arange(0, dim, 2, device=device, dtype=dtype)            # [dim/2]
        inv = 1.0 / (temperature ** (i / dim))
        ang = pos * inv                                                    # [L,dim/2]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)           # [L,dim]
        return emb

    y = get_1d_sincos(Hp, d_half)      # [Hp, D/2]
    x = get_1d_sincos(Wp, d_half)      # [Wp, D/2]
    pe = torch.zeros(Hp, Wp, D, device=device, dtype=dtype)
    pe[..., :d_half] = y[:, None, :]
    pe[..., d_half:] = x[None, :, :]
    return pe.reshape(1, Hp*Wp, D)

def build_spherical_sincos_pe(lon: torch.Tensor,
                              lat: torch.Tensor,
                              D: int,
                              device=None,
                              dtype=torch.float32,
                              temperature: float = 10000.0):
    """
    lon: [Hp, Wp] 경도 (radians, [-pi, pi])
    lat: [Hp, Wp] 위도 (radians, [-pi/2, pi/2])
    D: 임베딩 차원 (짝수)
    """
    assert D % 2 == 0, "pos dim D must be even"
    d_half = D // 2
    assert d_half % 2 == 0, "D/2 must be even"

    def encode_angle(angle: torch.Tensor, dim: int):
        """
        angle: [Hp, Wp]
        dim: 몇 차원으로 펼칠지
        """
        pos = angle.unsqueeze(-1)                          # [Hp, Wp, 1]
        i = torch.arange(0, dim, 2, device=device, dtype=dtype)  # [dim/2]
        inv = 1.0 / (temperature ** (i / dim))
        ang = pos * inv                                    # [Hp, Wp, dim/2]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [Hp, Wp, dim]
        return emb

    # 위도/경도 각각 D/2 차원
    emb_lat = encode_angle(lat.to(device=device, dtype=dtype), d_half)  # [Hp,Wp,D/2]
    emb_lon = encode_angle(lon.to(device=device, dtype=dtype), d_half)  # [Hp,Wp,D/2]

    pe = torch.cat([emb_lat, emb_lon], dim=-1)  # [Hp, Wp, D]
    return pe.reshape(1, -1, D)                 # [1, Hp*Wp, D]

def get_vae_spatial_factor(vae, default=8):
    try:
        conf = getattr(vae, "config", None)
        if conf and hasattr(conf, "block_out_channels"):
            return 2 ** (len(conf.block_out_channels) - 1)
    except Exception:
        pass
    return default

def make_timestep_1d(t, batch, device, *, float_for_dit=False):
    if torch.is_tensor(t):
        t1d = t.reshape(1).expand(batch).to(device)
    else:
        t1d = torch.full((batch,), t, device=device)
    return t1d.to(torch.bfloat16) if float_for_dit else t1d

# ------------------------------------------------------------------------
# NOTE : 타일 위치 시각화
# ------------------------------------------------------------------------

def dirs_to_equirect_uv(dirs_world, H, W, *, top_is_north: bool = True):
    # dirs_world: [N,3], normalized (x,y,z), 전방 +Z 컨벤션
    d = dirs_world / (torch.linalg.norm(dirs_world, dim=-1, keepdim=True) + 1e-8)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    lon = torch.atan2(x, z)                   # [-π, π]  (경도)
    lat = torch.asin(torch.clamp(y, -1, 1))   # [-π/2, π/2] (위도)

    U = (lon / (2*math.pi) + 0.5) * W
    if top_is_north:
        V = (0.5 - lat / math.pi) * H   # +90°(북극) -> V=0 (상단)
    else:
        V = (lat / math.pi + 0.5) * H   # +90°(북극) -> V=H (하단)

    return torch.stack([U, V], dim=-1)

@torch.no_grad()
def _hsv_color(i: int, N: int) -> tuple:
    """타일마다 다른 색상 (RGB 0~255)"""
    h = (i % max(N,1)) / max(N,1)
    s, v = 0.75, 1.0
    import colorsys
    r,g,b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r*255), int(g*255), int(b*255))

@torch.no_grad()
def _edge_uvs_for_tile(tile: dict, step: int, pano_H: int, pano_W: int) -> list:
    """
    타일의 4개 가장자리(Top/Right/Bottom/Left)를 step 간격으로 샘플해
    equirect uv(부동소수점) 리스트 4개를 반환.
    """
    H = int(tile["H"]); W = int(tile["W"])
    K = tile["K"]; R = tile["R"]
    dev = K.device; dt = K.dtype

    def rc_to_uv(rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        # rows/cols: [L]
        fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]
        u_n = (cols.to(dt) - cx) / fx
        v_n = (rows.to(dt) - cy) / fy
        d_cam = torch.stack([u_n, v_n, torch.ones_like(u_n)], dim=-1)  # [L,3]
        d_cam = d_cam / torch.linalg.norm(d_cam, dim=-1, keepdim=True)
        R_ = R.to(dtype=dt, device=dev)
        d_world = d_cam @ R_.transpose(0,1)                              # world<-cam
        return dirs_to_equirect_uv(d_world.reshape(-1,3),pano_H, pano_W, top_is_north=True)             # [L,2]

    xs = torch.arange(0, W, max(1, step), device=dev)
    ys = torch.arange(0, H, max(1, step), device=dev)

    # 상/하/좌/우 에지
    top_uv    = rc_to_uv(torch.zeros_like(xs), xs)
    bottom_uv = rc_to_uv(torch.full_like(xs, H-1), xs)
    left_uv   = rc_to_uv(ys, torch.zeros_like(ys))
    right_uv  = rc_to_uv(ys, torch.full_like(ys, W-1))

    return [top_uv, right_uv, bottom_uv.flip(0), left_uv.flip(0)]  # 시계방향으로 닫힘

@torch.no_grad()
def _draw_polyline_wrapped(draw: ImageDraw.ImageDraw, uvs: torch.Tensor, pano_W: int, color: tuple, width: int=2):
    """
    equirect의 수평 래핑을 고려해, u 차이가 크게 벌어지면 선분을 끊어서 그림.
    uvs: [L,2] (float)
    """
    if uvs.shape[0] < 2: return
    us = uvs[:,0].cpu().numpy()
    vs = uvs[:,1].cpu().numpy()
    path = [(float(us[0]), float(vs[0]))]
    for i in range(1, len(us)):
        if abs(us[i] - us[i-1]) > pano_W * 0.5:
            if len(path) >= 2:
                draw.line(path, fill=color, width=width)
            path = []
        path.append((float(us[i]), float(vs[i])))
    if len(path) >= 2:
        draw.line(path, fill=color, width=width)

@torch.no_grad()
def save_tiles_on_panorama(
    tiles: list,
    pano_H: int = 512,
    pano_W: int = 4096,
    *,
    mode: str = "outline",     # "outline" | "dots"
    step: int = 6,             # 샘플 간격(작을수록 촘촘)
    point_radius: int = 0,     # dots 모드에서 점 두께
    outpath: str = "tiles_on_panorama.png",
    annotate: bool = True,
    save_groups: bool = True,  # <- 추가: 10개 묶음 이미지도 저장할지
    group_size: int = 10       # <- 추가: 묶음 크기
):

    assert mode in ("outline", "dots"), "mode must be 'outline' or 'dots'"

    def _draw_subset(idx_list, out_path):
        canvas = Image.fromarray(np.zeros((pano_H, pano_W, 3), dtype=np.uint8))
        draw = ImageDraw.Draw(canvas)
        N_all = len(tiles)

        for idx in idx_list:
            tile = tiles[idx]
            Ht, Wt = int(tile["H"]), int(tile["W"])
            if Ht == 0 or Wt == 0:
                continue

            color = _hsv_color(idx, N_all)

            if mode == "outline":
                loops = _edge_uvs_for_tile(tile, step, pano_H, pano_W)
                for loop in loops:
                    uvs = torch.stack([
                        loop[:, 0].clamp(0, pano_W - 1),
                        loop[:, 1].clamp(0, pano_H - 1)
                    ], dim=-1)
                    _draw_polyline_wrapped(draw, uvs, pano_W, color=color, width=2)

                fx, fy = tile["K"][0, 0], tile["K"][1, 1]
                cx, cy = tile["K"][0, 2], tile["K"][1, 2]
                u_n = (torch.tensor([Wt / 2.0], device=tile["K"].device) - cx) / fx
                v_n = (torch.tensor([Ht / 2.0], device=tile["K"].device) - cy) / fy
                d_cam = torch.stack([u_n, v_n, torch.ones_like(u_n)], dim=-1)  # [1,3]
                d_cam = d_cam / torch.linalg.norm(d_cam, dim=-1, keepdim=True)
                d_world = d_cam @ tile["R"].transpose(0, 1)
                uv_ctr = dirs_to_equirect_uv(d_world, pano_H, pano_W, top_is_north=True)[0]
                u0 = int(uv_ctr[0].clamp(0, pano_W - 1).item())
                v0 = int(uv_ctr[1].clamp(0, pano_H - 1).item())
                # v0 = pano_H - 1 -v0
                draw.ellipse((u0 - 2, v0 - 2, u0 + 2, v0 + 2), outline=color, fill=color)
                if annotate and ("yaw" in tile and "pitch" in tile):
                    draw.text((u0 + 4, v0 + 4), f"{idx:02d}\ny:{tile['yaw']:.0f}\np:{tile['pitch']:.0f}", fill=color)

            elif mode == "dots":
                dev = tile["K"].device
                dt = torch.float32
                ys = torch.arange(0, Ht, max(1, step), device=dev, dtype=dt)
                xs = torch.arange(0, Wt, max(1, step), device=dev, dtype=dt)
                yy, xx = torch.meshgrid(ys, xs, indexing='ij')

                fx, fy = tile["K"][0, 0], tile["K"][1, 1]
                cx, cy = tile["K"][0, 2], tile["K"][1, 2]
                u_n = (xx - cx) / fx
                v_n = (yy - cy) / fy
                d_cam = torch.stack([u_n, v_n, torch.ones_like(u_n)], dim=-1)
                d_cam = d_cam / torch.linalg.norm(d_cam, dim=-1, keepdim=True)
                d_world = d_cam @ tile["R"].transpose(0, 1)
                uv = dirs_to_equirect_uv(d_world.reshape(-1, 3), pano_H, pano_W, top_is_north=True)
                u = uv[:, 0].clamp(0, pano_W - 1).round().long().cpu().numpy()
                v = uv[:, 1].clamp(0, pano_H - 1).round().long().cpu().numpy()
                # v = (pano_H-1-v).astype(int)

                arr = np.array(canvas)
                col = np.array(color, dtype=np.uint8)
                if point_radius <= 0:
                    arr[v, u] = col
                else:
                    for (uu, vv) in zip(u, v):
                        y0, y1 = max(0, vv - point_radius), min(pano_H, vv + point_radius + 1)
                        x0, x1 = max(0, uu - point_radius), min(pano_W, uu + point_radius + 1)
                        arr[y0:y1, x0:x1] = col
                canvas = Image.fromarray(arr)
                draw = ImageDraw.Draw(canvas)

        canvas.save(out_path)
        return out_path

    paths = []
    paths.append(_draw_subset(list(range(len(tiles))), outpath))

    if save_groups and group_size > 0:
        base, ext = os.path.splitext(outpath)
        group_id = 0
        for start in range(0, len(tiles), group_size):
            sub_idx = list(range(start, min(start + group_size, len(tiles))))
            out_path_g = f"{base}_{mode}_{group_id:02d}{ext}"
            paths.append(_draw_subset(sub_idx, out_path_g))
            group_id += 1

    return paths
