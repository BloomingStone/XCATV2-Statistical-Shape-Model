# %%
from pathlib import Path
import pyvista as pv
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_dilation
from skimage.morphology import ball
import nibabel as nib
from get_surface_cloud import get_cloud_from_nii_label
from align_surface import deform_surface

pv.set_jupyter_backend("html")

test_root_dir = Path("/media/data3/sj/Data/Phatom/test_ssm_image_ASOCA_144x144x128")
pca_root_dir = Path("/media/data3/sj/Data/Phatom/output_ssm_pca")

# %%
label_dir = test_root_dir / "label"
label_file = next(label_dir.glob("*.nii.gz"))

vtk_output_dir = test_root_dir / "vtk"
vtk_output_dir.mkdir(parents=True, exist_ok=True)

vtk_cloud = get_cloud_from_nii_label(label_file)

# %%
template_vtk_file = pca_root_dir / "ssm_template_avg.vtk"
template_cloud = pv.read(template_vtk_file)
landmark_cloud = deform_surface(
    source_surface=template_cloud,
    target_surface=vtk_cloud,
    alpha = 1e-5,
    beta = 1000,
    max_iterations = 5000,
    tolerance = 1e-6,
)
landmark_cloud.save(vtk_output_dir / "landmark.vtk")

# %%
plotter = pv.Plotter(shape=(1, 3))
plotter.subplot(0, 0)
plotter.add_mesh(vtk_cloud, point_size=5, scalars="label")
plotter.subplot(0, 1)
plotter.add_mesh(landmark_cloud, point_size=5, scalars="label")
plotter.subplot(0, 2)
plotter.add_mesh(template_cloud, point_size=5, scalars="label")
plotter.link_views()
plotter.show()


# %%
P_motion_file = pca_root_dir / "P_motion.npy"
b_motion_mean_per_phase_file = pca_root_dir / "b_motion_mean_per_phase.npy"
P_motion = np.load(P_motion_file)   # shape = (num_labels(L), num_components(N_m), num_points(M), 3)
b_motion_mean_per_phase = np.load(b_motion_mean_per_phase_file)  # shape = (num_labels(L), num_phases(N_j), num_components(N_m))

def get_deformed_points(
        landmark_cloud: pv.PolyData,
        P_motion: np.ndarray,
        b_motion_mean_per_phase: np.ndarray,
        phase: int,
        num_components_used: int | None = None,
) -> pv.PolyData:
    """
    计算通过运动变形后的点云
    Args:
        landmark_cloud (pv.PolyData): 特征点云， 每一个label中特征点数量相等
        P_motion (np.ndarray): 运动模式， shape = (num_labels(L), num_components(N_m), num_points(M), 3)
        b_motion_mean_per_phase (np.ndarray): 运动系数， shape = (num_labels(L), num_phases(N_j), num_components(N_m))
        phase (int): 当前相位, 范围[0, num_phases)
        num_components_used (int): 使用的运动模式数量
    Returns:
        deformed_points (pv.PolyData): 变形后的点云
    """
    deformed_points = pv.PolyData()
    num_labels = P_motion.shape[0]
    for label_id in range(num_labels):
        b = b_motion_mean_per_phase[label_id, phase]  # (N_m,)
        P = P_motion[label_id]  # (N_m, M, 3)
        
        if num_components_used is not None:
            b = b[:num_components_used]
            P = P[:num_components_used]
        
        deformation = np.tensordot(b, P, axes=([0], [0]))  # (M, 3)
        points = landmark_cloud.points[landmark_cloud.point_data['label'] == label_id+1]
        
        if len(points) != deformation.shape[0]:
            raise ValueError(f"Point count mismatch for label {label_id}")
        
        new_points = points + deformation  # (M, 3)
        deformed = pv.PolyData(new_points)
        deformed.point_data["label"] = np.ones(new_points.shape[0], dtype=np.uint8) * (label_id+1)
        deformed_points = deformed_points.merge(deformed)
    return deformed_points

def visualize_deformed_points_as_gif(
        polydata_list: list[pv.PolyData],
        gif_path: Path
    ) -> None:
    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif(str(gif_path))
    for polydata in polydata_list:
        plotter.clear()
        plotter.add_mesh(polydata, point_size=5, scalars="label")
        plotter.write_frame()
    
    plotter.close()
    print(f"GIF saved to {gif_path}")

polydata_list = []
for phase in range(b_motion_mean_per_phase.shape[1]):
    deformed_points = get_deformed_points(
        landmark_cloud=landmark_cloud,
        P_motion=P_motion,
        b_motion_mean_per_phase=b_motion_mean_per_phase,
        phase=phase,
    )
    polydata_list.append(deformed_points)

visualize_deformed_points_as_gif(
    polydata_list=polydata_list,
    gif_path=vtk_output_dir / "deformed_points.gif"
)

for phase, phase_polydata in enumerate(polydata_list):
    phase_polydata.save(vtk_output_dir / f"deformed_points_phase_{phase}.vtk")

        
# %%
def polydata_list_to_label_volumes(
    polydata_list: list[pv.PolyData],
    volume_shape: tuple,
    radius: int = 1,
    dilation_iter: int = 1,
    fill_holes: bool = True,
    save_dir: Path = None,
):
    """
    将多帧 polydata 点云（label 区分）转换为 label segmentation volumes。

    Args:
        polydata_list: list[pv.PolyData], 每帧一个，点为体素坐标系。
        volume_shape: tuple[int], 如原始图像 shape。
        radius: 插值球半径，控制每个点影响范围（默认为1）。
        dilation_iter: 膨胀迭代次数。
        fill_holes: 是否对每个 label 做空洞填充。
        save_dir: 如果指定，则保存为 label_phase_{id:02d}.nii.gz。
    Returns:
        List[np.ndarray]: 每帧对应的标签 volume。
    """
    label_volumes = []

    # 创建小球结构元素用于插值
    struct = ball(radius).astype(np.bool_)

    for i, poly in enumerate(polydata_list):
        label_volume = np.zeros(volume_shape, dtype=np.uint8)
        labels = np.unique(poly.point_data["label"])

        for label in labels:
            pts = poly.points[poly.point_data["label"] == label]
            pts = np.round(pts).astype(int)
            
            # 剔除边界外点
            valid = np.all((pts >= 0) & (pts < volume_shape), axis=1)
            pts = pts[valid]

            label_mask = np.zeros(volume_shape, dtype=bool)
            for pt in pts:
                z, y, x = pt  # 注意：Z-Y-X 顺序
                z_min, y_min, x_min = max(0, z - radius), max(0, y - radius), max(0, x - radius)
                z_max, y_max, x_max = min(volume_shape[0], z + radius + 1), min(volume_shape[1], y + radius + 1), min(volume_shape[2], x + radius + 1)
                label_mask[z_min:z_max, y_min:y_max, x_min:x_max] |= struct[
                    : z_max - z_min, : y_max - y_min, : x_max - x_min
                ]

            if fill_holes:
                label_mask = binary_fill_holes(label_mask.astype(np.bool_))

            for _ in range(dilation_iter):
                label_mask = binary_dilation(label_mask)

            label_volume[label_mask] = label

        label_volumes.append(label_volume)

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(label_volume, affine=np.eye(4)), save_dir / f"label_phase_{i:02d}.nii.gz")

    return 

label_image = nib.load(label_file)

label_volumes = polydata_list_to_label_volumes(
    polydata_list=polydata_list,
    volume_shape=label_image.shape,   # 替换为你原始标签图像的shape
    radius=5,
    dilation_iter=5,
    fill_holes=True,
    save_dir=test_root_dir / "label_phases",  # 保存路径
)
# %%
