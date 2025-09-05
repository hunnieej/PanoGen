import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------- ERP <-> angles helpers ----------
def degree_ticks(ax):
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90); ax.set_aspect(360/180/2)  # 2:1 aspect
    ax.set_xticks(np.linspace(0, 360, 9))
    ax.set_yticks(np.linspace(-90, 90, 7))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

def plot_wrapped(ax, lon_deg, lat_deg, lw=2.0):
    lon_deg = np.asarray(lon_deg)
    lat_deg = np.asarray(lat_deg)
    # split where we cross the 0/360 seam
    jumps = np.where(np.abs(np.diff(lon_deg)) > 180.0)[0]
    start = 0
    for j in list(jumps) + [len(lon_deg) - 1]:
        ax.plot(lon_deg[start:j+1], lat_deg[start:j+1], linewidth=lw)
        start = j + 1

def sph_from_lonlat(lon_deg, lat_deg):
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    return x, y, z

def erp_rect_edges(lon_c, lat_c, w_deg, h_deg, n=400):
    """Return edges (lon_deg, lat_deg) for top/right/bottom/left of an ERP rectangle."""
    lon_min = lon_c - w_deg/2.0
    lon_max = lon_c + w_deg/2.0
    lat_min = max(-90.0, lat_c - h_deg/2.0)
    lat_max = min(+90.0, lat_c + h_deg/2.0)
    # sample
    t = np.linspace(0.0, 1.0, n)
    # wrap longitudes into [0,360)
    def wrap(lon):
        return (lon + 360.0) % 360.0
    top_lon = wrap(lon_min + (lon_max - lon_min) * t);   top_lat = np.full_like(t, lat_max)
    bot_lon = wrap(lon_min + (lon_max - lon_min) * t);   bot_lat = np.full_like(t, lat_min)
    # for left/right edges vary lat
    left_lat = lat_min + (lat_max - lat_min) * t;        left_lon = wrap(np.full_like(t, lon_min))
    right_lat = lat_min + (lat_max - lat_min) * t;       right_lon = wrap(np.full_like(t, lon_max))
    return (top_lon, top_lat), (right_lon, right_lat), (bot_lon, bot_lat), (left_lon, left_lat)

def draw_erp_and_sphere(lon_c, lat_c, w_deg, h_deg, title, save_path):
    # 1) ERP figure
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1,2,1)
    degree_ticks(ax1)
    edges = erp_rect_edges(lon_c, lat_c, w_deg, h_deg, n=600)
    for lon, lat in edges:
        plot_wrapped(ax1, lon, lat, lw=2.0)
    ax1.set_title("ERP: rectangle on (lon, lat)")

    # 2) Sphere figure (3D)
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    # sphere grid (for context)
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(-np.pi/2, np.pi/2, 50)
    U,V = np.meshgrid(u,v)
    X = np.cos(V) * np.sin(U)
    Y = np.sin(V)
    Z = np.cos(V) * np.cos(U)
    ax2.plot_wireframe(X, Z, Y, rstride=6, cstride=6, linewidth=0.3, alpha=0.6)
    # draw edges projected onto sphere
    for lon, lat in edges:
        x,y,z = sph_from_lonlat(lon, lat)
        ax2.plot(x, z, y, linewidth=2.0)
    # view
    ax2.set_box_aspect([1,1,1])
    ax2.set_title("On unit sphere")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return save_path

# Example 1: equator-centered 60°x60°
p1 = draw_erp_and_sphere(180.0, 0.0, 60.0, 60.0,
                         title="ERP square 60°x60° at (lon=180°, lat=0°)",
                         save_path="./out/erp_square_to_sphere_equator_60x60.png")

# Example 2: near north pole 60°x60°
p2 = draw_erp_and_sphere(315.0, 15.0, 80.0, 80.0,
                         title="ERP square 60°x60° at (lon=315°, lat=45°)",
                         save_path="./out/erp_square_to_sphere_polar_60x60.png")

[p1, p2]
