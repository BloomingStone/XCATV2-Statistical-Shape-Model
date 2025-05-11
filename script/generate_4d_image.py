# %%
from pathlib import Path
import pyvista as pv
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_closing, generate_binary_structure
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
template_surface = pv.read(template_vtk_file)
landmark_surface = deform_surface(
    source_surface=template_surface,
    target_surface=vtk_cloud,
    device="cuda:1",
    alpha = 1e-5,
    beta = 1e-5,
    max_iterations = 5000,
    tolerance = 1e-6,
)
landmark_surface.save(vtk_output_dir / "landmark.vtk")

# %%
# calculate the zooming rate of template surface to the landmark surface
template_bounding_box = template_surface.bounds
landmark_bounding_box = landmark_surface.bounds
template_size = np.array([template_bounding_box[2*i+1] - template_bounding_box[2*i] for i in range(3)])
landmark_size = np.array([landmark_bounding_box[2*i+1] - landmark_bounding_box[2*i] for i in range(3)])
zooming_rate = np.mean(landmark_size / template_size)

# %%
plotter = pv.Plotter(shape=(1, 3))
plotter.subplot(0, 0)
plotter.add_mesh(vtk_cloud, point_size=5, scalars="label")
plotter.subplot(0, 1)
plotter.add_mesh(landmark_surface, point_size=5, scalars="label")
plotter.subplot(0, 2)
plotter.add_mesh(template_surface, point_size=5, scalars="label")
plotter.link_views()
plotter.show()


# %%
P_motion_file = pca_root_dir / "P_motion.npy"
b_motion_mean_per_phase_file = pca_root_dir / "b_motion_mean_per_phase.npy"
P_motion = np.load(P_motion_file)   # shape = (num_labels(L), num_components(N_m), num_points(M), 3)
b_motion_mean_per_phase = np.load(b_motion_mean_per_phase_file)  # shape = (num_labels(L), num_phases(N_j), num_components(N_m))

def get_deformed_surface(
        landmark_cloud: pv.PolyData,
        P_motion: np.ndarray,
        b_motion_mean_per_phase: np.ndarray,
        phase: int,
        zooming_rate: float = 1.0,
        num_components_used: int | None = None,
) -> pv.PolyData:
    """
    计算通过运动变形后的点云
    Args:
        landmark_cloud (pv.PolyData): 特征点云， 每一个label中特征点数量相等
        P_motion (np.ndarray): 运动模式， shape = (num_labels(L), num_components(N_m), num_points(M), 3)
        b_motion_mean_per_phase (np.ndarray): 运动系数， shape = (num_labels(L), num_phases(N_j), num_components(N_m)),
        zooming_rate (float): 运动系数缩放率，默认1.0
        phase (int): 当前相位, 范围[0, num_phases)
        num_components_used (int): 使用的运动模式数量
    Returns:
        deformed_points (pv.PolyData): 变形后的表面
    """
    deformed_points = pv.PolyData()
    num_labels = P_motion.shape[0]
    for label_id in range(num_labels):
        b = b_motion_mean_per_phase[label_id, phase]  # (N_m,)
        P = P_motion[label_id]  # (N_m, M, 3)
        b = b * zooming_rate  # (N_m,)
        
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
    res = landmark_cloud.copy()
    res.points = deformed_points.points
    return res

def visualize_deformed_points_as_gif(
        polydata_list: list[pv.PolyData],
        gif_path: Path
    ) -> None:
    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif(str(gif_path))
    for polydata in polydata_list:
        plotter.clear()
        plotter.add_mesh(polydata, scalars="label", opacity=0.5)
        plotter.write_frame()
    
    plotter.close()
    print(f"GIF saved to {gif_path}")

polydata_list = []
for phase in range(b_motion_mean_per_phase.shape[1]):
    motion_surface = get_deformed_surface(
        landmark_cloud=landmark_surface,
        P_motion=P_motion,
        b_motion_mean_per_phase=b_motion_mean_per_phase,
        zooming_rate=zooming_rate,
        phase=phase,
    )
    polydata_list.append(motion_surface)

visualize_deformed_points_as_gif(
    polydata_list=polydata_list,
    gif_path=vtk_output_dir / "deformed_points.gif"
)

for phase, phase_polydata in enumerate(polydata_list):
    phase_polydata.save(vtk_output_dir / f"deformed_points_phase_{phase}.vtk")

# %%
# TODO 最好是一开始从nii文件中提取surface时就给faces也标注上label
def identify_faces(polydata: pv.PolyData) -> pv.PolyData:
    """
    Identify faces in the polydata and assign labels to them.
    Args:
        polydata (pv.PolyData): Input polydata with point data "label".
    Returns:
        pv.PolyData: Polydata with face labels.
    """
    labels = polydata.point_data["label"]
    faces = polydata.faces.reshape(-1, 4)[:, 1:]
    face_labels = np.array([
        labels[faces[i, 0]] if labels[faces[i, 0]] == labels[faces[i, 1]] == labels[faces[i, 2]] else 0
        for i in range(faces.shape[0])
    ])
    res= polydata.copy()
    res.cell_data["label"] = face_labels
    return res

def extract_faces_by_label(polydata: pv.PolyData, label: int) -> pv.PolyData:
    """
    Extract faces from the polydata based on the specified label.
    Args:
        polydata (pv.PolyData): Input polydata with cell data "label".
        label (int): The label to extract.
    Returns:
        pv.PolyData: Extracted faces with the specified label.
    """
    labels = polydata.cell_data["label"]
    unique_labels = np.unique(labels)
    if label not in unique_labels:
        raise ValueError(f"Label {label} not found in polydata.")
    face_indices = np.where(labels == label)[0]
    return polydata.extract_cells(face_indices).extract_surface()

        
# %%
def polydata_to_label_volumes(
    polydata: pv.PolyData,
    output_shape: tuple[int, int, int], 
) -> np.ndarray :
    """
    将 polydata surface（以 label 区分）转换为 label segmentation volumes。

    Args:
        polydata: pv.PolyData, 输入的 polydata surface。
        output_shape: tuple[int, int, int], 体素坐标系的形状。
    Returns:
        label_volume: np.ndarray, 体素坐标系的标签体积。
    """
    reference_volume = pv.ImageData(dimensions=output_shape)
    label_volume = np.zeros(output_shape, dtype=np.uint8)
    labels = np.unique(polydata.point_data["label"]).tolist()
    labels = [int(label) for label in labels if label > 0 and label != 2]
    labels = sorted(labels, reverse=True)
    labels.append(2)   # 左心室部分 label=2， 因为此前和左心肌合并处理，故此处需要最后处理，以覆盖在所有label之上
    polydata = identify_faces(polydata)

    for label in labels:
        face_of_label = extract_faces_by_label(polydata, label)
        mask_image_data = face_of_label.voxelize_binary_mask(reference_volume = reference_volume)
        mask = mask_image_data.points[mask_image_data.point_data["mask"] == 1].astype(np.uint8)   # shape = (N, 3)
        mask_image = np.zeros(output_shape, dtype=np.uint8)
        mask_image[mask[:, 0], mask[:, 1], mask[:, 2]] = 1
        mask_image = binary_closing(mask_image, structure=generate_binary_structure(3, 3), iterations=1)
        mask_image = binary_fill_holes(mask_image, structure=generate_binary_structure(3, 3))
        label_volume[mask_image > 0] = label
    
    return label_volume

def save_label_volume_as_nii(
    label_volume: np.ndarray,
    output_file: Path,
    affine: np.ndarray,
    direction = (-1, 1, 1),
) -> None:
    """
    将标签体积保存为 NII 文件。
    
    Args:
        label_volumes: np.ndarray, 标签体积。
        output_file: Path, 输出文件路径。
        affine: np.ndarray | None, 仿射矩阵
        direction: tuple[int, int, int], 中间步骤处理时的方向方向，默认为 (-1, 1, 1)。
    """
    direction_label = np.diag(affine[:3, :3])
    for i in range(3):
        if direction[i] * direction_label[i] < 0:
            label_volume = np.flip(label_volume, axis=i)
    new_label_image = nib.Nifti1Image(label_volume, affine=nib.load(label_file).affine)
    nib.save(new_label_image, str(output_file))
    print(f"Saved {output_file}")


label_image = nib.load(label_file)
ref_image = pv.ImageData(dimensions=label_image.shape)
label_volumes = [ 
    polydata_to_label_volumes(
        polydata=polydata,
        output_shape=label_image.shape
    ) for polydata in polydata_list
]

for i in range(len(label_volumes)):
    save_label_volume_as_nii(
        label_volume=label_volumes[i],
        output_file=test_root_dir / f"label_volumes/phase_{i}.nii.gz",
        affine=label_image.affine,
    )

landmark_label_volume = polydata_to_label_volumes(
    polydata=landmark_surface,
    output_shape=label_image.shape
)
save_label_volume_as_nii(
    label_volume=landmark_label_volume,
    output_file=test_root_dir / "label_volumes/landmark.nii.gz",
    affine=label_image.affine,
)

# %%