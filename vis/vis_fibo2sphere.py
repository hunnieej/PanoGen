import numpy as np
import torch
import struct
import os
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def generate_fibonacci_lattice(N):
    """
    주어진 Fibonacci lattice 생성 함수
    """
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

def save_points_to_ply(points: torch.Tensor, 
                      filename: str, 
                      colors: Optional[torch.Tensor] = None,
                      point_size: float = 0.01,
                      use_spheres: bool = False) -> None:
    """
    점들을 PLY 파일로 저장
    
    Args:
        points: [N, 3] 점 좌표
        filename: 저장할 파일명
        colors: [N, 3] RGB 색상 (0-255), None이면 Z 좌표 기반 색상
        point_size: 점 크기 (구체 반지름)
        use_spheres: True면 각 점을 작은 구체로 표현
    """
    points_np = points.numpy()
    N = len(points_np)
    
    # 색상 생성
    if colors is None:
        # Z 좌표를 기반으로 색상 생성 (viridis 색상맵)
        z_normalized = (points_np[:, 2] + 1) / 2  # -1~1 -> 0~1
        colormap = cm.get_cmap('viridis')
        colors_rgba = colormap(z_normalized)
        colors_rgb = (colors_rgba[:, :3] * 255).astype(np.uint8)
    else:
        colors_rgb = colors.numpy().astype(np.uint8)
    
    # PLY 헤더 작성
    header = f"""ply
format ascii 1.0
comment Fibonacci Lattice Points (N={N})
comment Generated from golden ratio spiral
element vertex {N}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
"""
    
    if use_spheres:
        # 각 점을 작은 구체로 표현하기 위한 면 추가
        sphere_faces = generate_sphere_faces(point_size)
        num_faces = len(sphere_faces) * N
        header += f"""element face {num_faces}
property list uchar int vertex_indices
"""
    
    header += "end_header\n"
    
    # 파일 저장
    with open(filename, 'w') as f:
        f.write(header)
        
        # 점 데이터 작성
        for i in range(N):
            x, y, z = points_np[i]
            r, g, b = colors_rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        
        # 구체 면 데이터 (선택적)
        if use_spheres:
            for i in range(N):
                center = points_np[i]
                for face in sphere_faces:
                    f.write(f"3 {face[0]+i*8} {face[1]+i*8} {face[2]+i*8}\n")
    
    print(f"[INFO] PLY file saved: {filename}")
    print(f"[INFO] Points: {N}, File size: {os.path.getsize(filename)/1024:.2f} KB")

def generate_sphere_faces(radius: float) -> List[Tuple[int, int, int]]:
    """간단한 구체 면 생성 (정육면체 기반)"""
    # 간단한 정육면체 면 (각 점당 8개 정점, 12개 면)
    faces = [
        (0, 1, 2), (0, 2, 3),  # 앞면
        (4, 7, 6), (4, 6, 5),  # 뒷면
        (0, 4, 5), (0, 5, 1),  # 왼쪽면
        (2, 6, 7), (2, 7, 3),  # 오른쪽면
        (0, 3, 7), (0, 7, 4),  # 아래면
        (1, 5, 6), (1, 6, 2)   # 위면
    ]
    return faces

def save_wireframe_sphere_ply(points: torch.Tensor, 
                             filename: str,
                             radius: float = 1.0,
                             resolution: int = 50) -> None:
    """
    Fibonacci 점들과 함께 wireframe 구체를 PLY로 저장
    """
    points_np = points.numpy()
    N = len(points_np)
    
    # 구체 wireframe 생성
    phi_range = np.linspace(0, np.pi, resolution)
    theta_range = np.linspace(0, 2*np.pi, resolution)
    
    sphere_points = []
    sphere_lines = []
    
    # 경선 (meridians)
    for i, theta in enumerate(theta_range[:-1]):  # 마지막 점 제외 (중복)
        for j, phi in enumerate(phi_range):
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            sphere_points.append([x, y, z])
            
            # 선 연결 (세로)
            if j < len(phi_range) - 1:
                sphere_lines.append([len(sphere_points)-1, len(sphere_points)])
    
    # 위선 (parallels)
    for j, phi in enumerate(phi_range[1:-1], 1):  # 극점 제외
        for i, theta in enumerate(theta_range[:-1]):
            current_idx = j * (len(theta_range)-1) + i
            next_idx = j * (len(theta_range)-1) + ((i+1) % (len(theta_range)-1))
            sphere_lines.append([current_idx, next_idx])
    
    sphere_points = np.array(sphere_points)
    total_points = N + len(sphere_points)
    
    # PLY 헤더
    header = f"""ply
format ascii 1.0
comment Fibonacci Lattice with Wireframe Sphere
element vertex {total_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element edge {len(sphere_lines)}
property int vertex1
property int vertex2
end_header
"""
    
    # 색상 설정
    z_normalized = (points_np[:, 2] + 1) / 2
    colormap = cm.get_cmap('viridis')
    fib_colors = (colormap(z_normalized)[:, :3] * 255).astype(np.uint8)
    sphere_colors = np.full((len(sphere_points), 3), [128, 128, 128], dtype=np.uint8)  # 회색
    
    # 파일 저장
    with open(filename, 'w') as f:
        f.write(header)
        
        # Fibonacci 점들
        for i in range(N):
            x, y, z = points_np[i]
            r, g, b = fib_colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        
        # 구체 wireframe 점들
        for i, (x, y, z) in enumerate(sphere_points):
            r, g, b = sphere_colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        
        # 구체 wireframe 선들
        for line in sphere_lines:
            v1, v2 = line[0] + N, line[1] + N  # Fibonacci 점들 이후 인덱스
            f.write(f"{v1} {v2}\n")
    
    print(f"[INFO] Wireframe sphere PLY saved: {filename}")

def create_colored_fibonacci_ply(points: torch.Tensor, 
                                filename: str,
                                color_mode: str = 'height') -> None:
    """
    다양한 색상 모드로 Fibonacci lattice PLY 생성
    
    Args:
        color_mode: 'height', 'index', 'distance', 'spiral'
    """
    points_np = points.numpy()
    N = len(points_np)
    
    # 색상 모드별 색상 생성
    if color_mode == 'height':
        # Z 좌표 기반 (높이)
        values = (points_np[:, 2] + 1) / 2  # -1~1 -> 0~1
        colormap = cm.get_cmap('viridis')
        
    elif color_mode == 'index':
        # 인덱스 기반 (나선 순서)
        values = np.arange(N) / N
        colormap = cm.get_cmap('rainbow')
        
    elif color_mode == 'distance':
        # 북극점으로부터의 거리
        north_pole = np.array([0, 0, 1])
        distances = np.linalg.norm(points_np - north_pole, axis=1)
        values = distances / distances.max()
        colormap = cm.get_cmap('plasma')
        
    elif color_mode == 'spiral':
        # 나선 각도 기반
        phi = (1 + np.sqrt(5)) / 2
        indices = np.arange(N)
        theta = 2 * np.pi * indices / phi
        values = (theta % (2 * np.pi)) / (2 * np.pi)
        colormap = cm.get_cmap('hsv')
    
    else:
        raise ValueError(f"Unknown color mode: {color_mode}")
    
    colors_rgba = colormap(values)
    colors_rgb = (colors_rgba[:, :3] * 255).astype(np.uint8)
    
    # PLY 파일 저장
    save_points_to_ply(points, filename, torch.from_numpy(colors_rgb))
    print(f"[INFO] Color mode '{color_mode}' applied")

def analyze_and_save_fibonacci_distribution(N: int, 
                                          output_dir: str = "./out/fibonacci_ply_output") -> str:
    """
    Fibonacci lattice 분석하고 다양한 PLY 파일 생성
    """
    # 출력 디렉터리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Fibonacci Lattice 분석 및 PLY 생성 (N={N}) ===")
    
    # Fibonacci lattice 생성
    points = generate_fibonacci_lattice(N)
    
    # 기본 통계
    norms = torch.norm(points, dim=1)
    print(f"점들의 norm 범위: [{norms.min():.6f}, {norms.max():.6f}]")
    print(f"평균 norm: {norms.mean():.6f}")
    
    # 거리 분석 (샘플링)
    sample_size = min(500, N)
    sample_indices = torch.randperm(N)[:sample_size]
    sample_points = points[sample_indices]
    distances = torch.cdist(sample_points, sample_points)
    distances = distances[distances > 0]
    
    print(f"점들 간 거리 통계 (샘플 {sample_size}개):")
    print(f"  최소: {distances.min():.4f}")
    print(f"  최대: {distances.max():.4f}") 
    print(f"  평균: {distances.mean():.4f}")
    
    # 다양한 PLY 파일 생성
    base_path = os.path.join(output_dir, f"fibonacci_N{N}")
    
    # 1. 기본 높이 색상
    create_colored_fibonacci_ply(points, f"{base_path}_height.ply", 'height')
    
    # 2. 인덱스 색상 (나선 순서)
    create_colored_fibonacci_ply(points, f"{base_path}_index.ply", 'index')
    
    # 3. 거리 색상
    create_colored_fibonacci_ply(points, f"{base_path}_distance.ply", 'distance')
    
    # 4. 나선 각도 색상
    create_colored_fibonacci_ply(points, f"{base_path}_spiral.ply", 'spiral')
    
    # 5. Wireframe과 함께
    save_wireframe_sphere_ply(points, f"{base_path}_with_sphere.ply")
    
    # 시각화 및 저장
    visualize_and_save_analysis(points, output_dir, N)
    
    print(f"\n[완료] 모든 PLY 파일이 '{output_dir}' 디렉터리에 생성되었습니다.")
    return output_dir

def visualize_and_save_analysis(points: torch.Tensor, output_dir: str, N: int) -> None:
    """
    분석 결과 시각화 및 이미지 저장
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Fibonacci Lattice Analysis (N={N})', fontsize=16)
    
    points_np = points.numpy()
    
    # 1. 3D 시각화
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter1 = ax1.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2],
                          c=points_np[:, 2], cmap='viridis', s=20, alpha=0.8)
    ax1.set_title('3D View (Height Colored)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.colorbar(scatter1, ax=ax1, shrink=0.6)
    
    # 2. XY 투영
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(points_np[:, 0], points_np[:, 1],
                          c=range(N), cmap='rainbow', s=15, alpha=0.7)
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--')
    ax2.add_patch(circle)
    ax2.set_title('XY Projection (Index Colored)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 3. 거리 분포
    ax3 = axes[0, 2]
    sample_size = min(300, N)
    sample_indices = torch.randperm(N)[:sample_size]
    sample_points = points[sample_indices]
    distances = torch.cdist(sample_points, sample_points)
    distances = distances[distances > 0].numpy()
    
    ax3.hist(distances, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(distances.mean(), color='red', linestyle='--',
                label=f'Mean: {distances.mean():.3f}')
    ax3.set_title('Distance Distribution')
    ax3.set_xlabel('Distance')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Z 분포
    ax4 = axes[1, 0]
    ax4.hist(points_np[:, 2], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax4.axhline(N/20, color='red', linestyle='--', label=f'Uniform: {N/20:.1f}')
    ax4.set_title('Z Distribution')
    ax4.set_xlabel('Z Coordinate')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 나선 패턴
    ax5 = axes[1, 1]
    phi = (1 + np.sqrt(5)) / 2
    indices = np.arange(N)
    theta = 2 * np.pi * indices / phi
    ax5.plot(indices, theta % (2*np.pi), 'b.', markersize=2, alpha=0.7)
    ax5.set_title('Spiral Pattern')
    ax5.set_xlabel('Point Index')
    ax5.set_ylabel('Theta (mod 2π)')
    ax5.grid(True, alpha=0.3)
    
    # 6. 색상 범례
    ax6 = axes[1, 2]
    ax6.axis('off')
    info_text = f"""
PLY Files Generated:
• fibonacci_N{N}_height.ply (Z-colored)
• fibonacci_N{N}_index.ply (Order-colored)  
• fibonacci_N{N}_distance.ply (Distance-colored)
• fibonacci_N{N}_spiral.ply (Angle-colored)
• fibonacci_N{N}_with_sphere.ply (With wireframe)

Statistics:
• Total points: {N}
• Coordinate range: [{points_np.min():.3f}, {points_np.max():.3f}]
• Distance range: [{distances.min():.4f}, {distances.max():.4f}]
• Memory usage: {points.element_size() * points.nelement() / 1024:.2f} KB

Visualization:
Open PLY files in MeshLab, CloudCompare, 
Blender, or other 3D viewers to explore
the spherical distribution interactively.
"""
    ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # 이미지 저장
    image_path = os.path.join(output_dir, f"fibonacci_N{N}_analysis.png")
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"[INFO] Analysis image saved: {image_path}")

def main():
    """메인 실행 함수"""
    print("Fibonacci Lattice PLY Generator")
    print("=" * 50)
    
    # 다양한 크기로 PLY 파일 생성
    sizes = [2600, 6000, 10000]
    
    for N in sizes:
        print(f"\n{'='*20} N = {N} {'='*20}")
        output_dir = analyze_and_save_fibonacci_distribution(N)
    
    print(f"\n{'='*50}")

if __name__ == "__main__":
    main()