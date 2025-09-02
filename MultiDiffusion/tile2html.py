import torch, numpy as np, os
import plotly.graph_objects as go
from projection import generate_fibonacci_lattice, spherical_to_perspective_tiles, make_view_centers

# helpers (same as before)
def tile_edge_dirs_world(tile: dict, step: int = 8):
    H = int(tile["H"]); W = int(tile["W"])
    K = tile["K"]; R = tile["R"]
    dev = K.device
    dt = torch.float32
    xs = torch.arange(0, W, max(1, step), device=dev, dtype=dt)
    ys = torch.arange(0, H, max(1, step), device=dev, dtype=dt)
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    def rc_to_dir(rows, cols):
        u_n = (cols - cx) / fx
        v_n = (rows - cy) / fy
        d_cam = torch.stack([u_n, v_n, torch.ones_like(u_n)], dim=-1)
        d_cam = d_cam / torch.linalg.norm(d_cam, dim=-1, keepdim=True)
        d_world = d_cam @ R.transpose(0,1)
        d_world = d_world / torch.linalg.norm(d_world, dim=-1, keepdim=True)
        return d_world
    top    = rc_to_dir(torch.zeros_like(xs), xs)
    bottom = rc_to_dir(torch.full_like(xs, H-1), xs)
    left   = rc_to_dir(ys, torch.zeros_like(ys))
    right  = rc_to_dir(ys, torch.full_like(ys, W-1))
    return [top, right, bottom.flip(0), left.flip(0)]

def tile_center_dir_world(tile: dict):
    H = int(tile["H"]); W = int(tile["W"])
    K = tile["K"]; R = tile["R"]
    cx, cy = K[0,2], K[1,2]
    fx, fy = K[0,0], K[1,1]
    u_n = (torch.tensor([W/2.0], device=K.device) - cx) / fx
    v_n = (torch.tensor([H/2.0], device=K.device) - cy) / fy
    d_cam = torch.stack([u_n, v_n, torch.ones_like(u_n)], dim=-1)
    d_cam = d_cam / torch.linalg.norm(d_cam, dim=-1, keepdim=True)
    d_world = (d_cam @ R.transpose(0,1))[0]
    d_world = d_world / torch.linalg.norm(d_world)
    return d_world

# build tiles (two patterns: 89 fixed vs overlap-based)
dirs = generate_fibonacci_lattice(2600).to(torch.float32)
tiles_89 = spherical_to_perspective_tiles(dirs=dirs, H=64, W=64, fov_deg=80.0, overlap=0.6)

# alternate pattern
yaws, pitches = make_view_centers(fov_deg=80.0, overlap=0.6)
tiles_overlap = []
K = tiles_89[0]["K"]
for yaw_deg, pitch_deg in zip(yaws, pitches):
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    look_dir = torch.tensor([np.cos(pitch)*np.cos(yaw),
                             np.sin(pitch),
                             np.cos(pitch)*np.sin(yaw)], dtype=dirs.dtype)
    R = tiles_89[0]["R"]  # reuse structure, approximate extrinsic
    tiles_overlap.append({"yaw":yaw_deg,"pitch":pitch_deg,"R":R,"K":K,"H":64,"W":64,
                          "used_idx":tiles_89[0]["used_idx"],
                          "lin_coords":tiles_89[0]["lin_coords"]})

# function to make traces for given tiles, group_size, show_labels
def make_group_traces(tiles, group_size=10, show_labels=True, name_prefix=""):
    traces = []
    num_groups = (len(tiles) + group_size - 1) // group_size
    for g in range(num_groups):
        start, end = g*group_size, min((g+1)*group_size, len(tiles))
        Xe,Ye,Ze,Xc,Yc,Zc,texts=[],[],[],[],[],[],[]
        for idx in range(start,end):
            tile=tiles[idx]
            edges = tile_edge_dirs_world(tile, step=6)
            for poly in edges:
                Xe += poly[:,0].tolist()+[None]
                Ye += poly[:,1].tolist()+[None]
                Ze += poly[:,2].tolist()+[None]
            ctr = tile_center_dir_world(tile).cpu().numpy()
            Xc.append(ctr[0]); Yc.append(ctr[1]); Zc.append(ctr[2])
            texts.append(f"{idx}" if show_labels else "")
        edge_tr = go.Scatter3d(x=Xe,y=Ye,z=Ze,mode="lines",
                               line=dict(width=3),
                               name=f"{name_prefix}Edges {start}-{end-1}",visible=False)
        ctr_tr = go.Scatter3d(x=Xc,y=Yc,z=Zc,mode="markers+text",
                              marker=dict(size=3),
                              text=texts,textposition="top center",
                              name=f"{name_prefix}Centers {start}-{end-1}",visible=False)
        traces.extend([edge_tr, ctr_tr])
    return traces, num_groups

# sphere mesh
u = np.linspace(0,2*np.pi,80); v = np.linspace(0,np.pi,40)
xs = np.outer(np.cos(u), np.sin(v)); ys = np.outer(np.sin(u), np.sin(v)); zs = np.outer(np.ones_like(u), np.cos(v))
surface = go.Surface(x=xs,y=ys,z=zs,showscale=False,opacity=0.15,colorscale="Greys")

# base traces (sphere + groups for both patterns)
traces = [surface]
tr_89, num_g89 = make_group_traces(tiles_89, group_size=10, show_labels=True, name_prefix="89-")
tr_ov, num_gov = make_group_traces(tiles_overlap, group_size=10, show_labels=True, name_prefix="Overlap-")
traces += tr_89 + tr_ov

# default visible: first group of 89
if len(tr_89)>=2:
    tr_89[0].visible=True; tr_89[1].visible=True

# buttons for groups
buttons=[]
for g in range(num_g89):
    vis=[True]+[False]*(len(tr_89)+len(tr_ov))
    vis[1+2*g]=True; vis[1+2*g+1]=True
    buttons.append(dict(label=f"89 {g*10}-{min((g+1)*10-1,len(tiles_89)-1)}",
                        method="update",args=[{"visible":vis}]))
for g in range(num_gov):
    vis=[True]+[False]*(len(tr_89)+len(tr_ov))
    base=len(tr_89)
    vis[1+base+2*g]=True; vis[1+base+2*g+1]=True
    buttons.append(dict(label=f"Overlap {g*10}-{min((g+1)*10-1,len(tiles_overlap)-1)}",
                        method="update",args=[{"visible":vis}]))

# label toggle
buttons_labels=[
    dict(label="Show labels",method="update",
         args=[{"text":[tr.text if isinstance(tr,go.Scatter3d) else None for tr in traces]}]),
    dict(label="Hide labels",method="update",
         args=[{"text":[[""]*len(tr.x) if isinstance(tr,go.Scatter3d) else None for tr in traces]}])
]

fig = go.Figure(data=traces)
fig.update_layout(
    title="SphereDiff Tiles — Interactive Viewer",
    scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),
               aspectmode="data"),
    updatemenus=[
        dict(type="buttons",direction="down",x=0,y=1.15,buttons=buttons),
        dict(type="buttons",direction="right",x=0,y=1.05,buttons=buttons_labels)
    ]
)

html_path = "/mnt/data/tiles_interactive_custom.html"
fig.write_html(html_path, include_plotlyjs="cdn")
html_path
