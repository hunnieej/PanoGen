# Make an interactive, rotatable sphere textured with the given ERP image.
# Output: an HTML you can open in a browser and drag to rotate.

import numpy as np
from PIL import Image
import plotly.graph_objects as go

# Load ERP image the user provided
erp_path = "./outputs/test/t5_firefly_2.1/final.png"  # equirectangular panorama (W=4096, H=512 typically)
img = Image.open(erp_path).convert("RGB")
W, H = img.size

# Build a sphere mesh and sample the ERP as texture (vertex colors)
# Choose a moderate grid resolution to keep HTML lightweight
nu, nv = 256, 128  # longitude, latitude samples
u = np.linspace(0, 2*np.pi, nu, endpoint=False)    # longitude
v = np.linspace(0, np.pi, nv)                      # latitude

uu, vv = np.meshgrid(u, v)  # [nv, nu]

# Sphere coordinates
x = np.cos(uu) * np.sin(vv)
y = np.sin(uu) * np.sin(vv)
z = np.cos(vv)

# Map sphere -> ERP pixel coords
U_px = (uu / (2*np.pi)) * (W - 1)
V_px = (vv / np.pi) * (H - 1)

# Sample colors with nearest-neighbor (fast, sufficient for preview)
img_np = np.array(img)
Ui = np.clip(np.round(U_px).astype(int), 0, W-1)
Vi = np.clip(np.round(V_px).astype(int), 0, H-1)
colors = img_np[Vi, Ui]  # [nv,nu,3] in uint8

# Build triangles for Mesh3d
# Grid indexing -> two triangles per cell
verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])  # [nv*nu,3]
# Triangle indices
def cell_tris(i, j):
    a = i*nu + j
    b = i*nu + (j+1)%nu  # wrap in longitude
    c = (i+1)*nu + j
    d = (i+1)*nu + (j+1)%nu
    return [(a,b,c), (b,d,c)]

tri_i, tri_j, tri_k = [], [], []
for i in range(nv-1):
    for j in range(nu):
        t1, t2 = cell_tris(i, j)
        tri_i.append(t1[0]); tri_j.append(t1[1]); tri_k.append(t1[2])
        tri_i.append(t2[0]); tri_j.append(t2[1]); tri_k.append(t2[2])

# Vertex colors
vertexcolor = colors.reshape(-1, 3) / 255.0  # to 0..1 floats

mesh = go.Mesh3d(
    x=verts[:,0], y=verts[:,1], z=verts[:,2],
    i=tri_i, j=tri_j, k=tri_k,
    vertexcolor=vertexcolor,
    lighting=dict(ambient=1.0),  # no shading to keep true colors
    flatshading=True,
    name="ERP Textured Sphere",
    showscale=False
)

fig = go.Figure(data=[mesh])
fig.update_layout(
    title="Interactive ERP → Sphere (drag to rotate)",
    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
               aspectmode="data"),
    margin=dict(l=0, r=0, t=30, b=0)
)

html_path = "./outputs/test/t5_firefly_2.1/erp_sphere_interactive.html"
fig.write_html(html_path, include_plotlyjs="cdn")
html_path
