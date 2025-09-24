from pathlib import Path
from collections import defaultdict
from typing import Annotated


import pyvista as pv
import numpy as np
from sklearn.decomposition import PCA
import einops
import typer

from . import project_root
from .entrypoint import app


def load_vtk_points(base_folder: Path) -> np.ndarray:
    """
    Load VTK files and extract points for each label.
    Args:
        base_folder (Path): Path to the folder containing VTK files.
    Returns:
        np.ndarray: A 5D numpy array with shape (num_labels(L), num_patients(N_i), num_phases(N_j), num_points(M), 3).
    """
    data = defaultdict(lambda: defaultdict(dict))
    num_labels: None | int = None
    num_patients: int = 0
    num_phases: None | int = None
    num_points: None | int = None

    def process_patient_dir(patient_dir: Path) -> bool:
        nonlocal num_labels, num_phases, num_points, data
        num_phases_inner = 0
        for phase_id, phase_file in enumerate(sorted(patient_dir.iterdir())):
            if not phase_file.suffix == '.vtk':
                continue
            mesh = pv.read(phase_file)
            labels = mesh.point_data['label']
            all_labels = np.unique(labels)
            if num_labels is None:
                num_labels = len(all_labels)
            elif num_labels != len(all_labels):
                print(f"Error: Inconsistent number of labels in {patient_dir}")
                return False
            for label in all_labels:
                points = mesh.points[labels == label]
                if num_points is None:
                    num_points = points.shape[0]
                elif num_points != points.shape[0]:
                    print(f"Error: Inconsistent number of points in {phase_file}")
                    return False
                data[label][patient_dir.name][phase_id] = points
            num_phases_inner += 1
        if num_phases is None:
            num_phases = num_phases_inner
        elif num_phases != num_phases_inner:
            print(f"Error: Inconsistent number of phases in {patient_dir}")
            for label in all_labels:
                del data[label][patient_dir.name]
            return False

        return True


    for patient in sorted(base_folder.iterdir()):
        if not patient.is_dir():
            continue
        if not process_patient_dir(patient):
            continue

        num_patients += 1
    
    if num_labels is None or num_phases is None or num_points is None:
        raise ValueError("No valid VTK files found in the specified directory.")
    
    res = np.zeros((num_labels, num_patients, num_phases, num_points, 3), dtype=np.float32)
    
    for label_id, patients in enumerate(data.values()):
        for patient_id, phases in enumerate(patients.values()):
            for phase_id, point in enumerate(phases.values()):
                res[label_id][patient_id][phase_id] = point
    
    return res


def calculate_mean_points_per_patient(points: np.ndarray) -> np.ndarray:
    """
    Calculate the mean patient points for each label and patient across all phases.
    Args:
        points (np.ndarray): The input points array, shape = (num_labels(L), num_patients(N_i), num_phases(N_j), 3, num_points(M)).
    Returns:
        np.ndarray: The mean patient points, shape = (num_labels(L), num_patients(N_i), num_points(M), 3).
    """
    assert points.ndim == 5, f"points.ndim is {points.ndim}, expected 5"
    assert points.shape[4] == 3, f"points.shape[4] is {points.shape[4]}, expected 3"
    return np.mean(points, axis=2)


def calculate_mean_points_per_phase(points: np.ndarray) -> np.ndarray:
    """
    Calculate the mean points for each label and phase across all patients.
    Args:
        points (np.ndarray): The input points array, shape = (num_labels(L), num_patients(N_i), num_phases(N_j), num_points(M), 3).
    Returns:
        np.ndarray: The mean points, shape = (num_labels(L), num_phases(N_j), num_points(M), 3).
    """
    assert points.ndim == 5, f"points.ndim is {points.ndim}, expected 5"
    assert points.shape[4] == 3, f"points.shape[4] is {points.shape[4]}, expected 3"
    return np.mean(points, axis=1)

def calculate_mean_points(points: np.ndarray) -> np.ndarray:
    """
    Calculate the mean points for each label across all patients and phases.
    Args:
        points (np.ndarray): The input points array, shape = (num_labels(L), num_patients(N_i), num_phases(N_j), num_points(M), 3).
    Returns:
        np.ndarray: The mean points, shape = (num_labels(L), num_points(M), 3).
    """
    assert points.ndim == 5, f"points.ndim is {points.ndim}, expected 5"
    assert points.shape[4] == 3, f"points.shape[4] is {points.shape[4]}, expected 3"
    return np.mean(points, axis=(1, 2))


def calculate_anatomy_pca(s_mean_patients: np.ndarray, s_mean_global: np.ndarray, N_a: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform PCA on the mean points and calculate the PCA components.
    Args:
        s_mean_points (np.ndarray): The mean points array, shape = (num_labels(L), num_patients(N_i), num_points(M), 3).
        s_mean_global (np.ndarray): The mean global points array, shape = (num_labels(L), num_points(M), 3).
        N_a (int, optional): Number of anatomy components. Defaults to 7. And the final components will be min(N_i, N_a) where N_i is number of samples and N_a is number of featrues
    Returns:
        Principal component direction, shape = (num_labels(L), num_components(N_a), num_points(M), 3)
        Principal component weights, shape = (num_labels(L), num_patients(N_i), num_components(N_a))
        Explained variance(lambda), shape = (num_labels(L), num_components(N_a))
    """
    assert s_mean_patients.ndim == 4, f"s_mean_points.ndim is {s_mean_patients.ndim}, expected 4"
    assert s_mean_global.ndim == 3, f"s_mean_global.ndim is {s_mean_global.ndim}, expected 3"
    
    
    L, N_i, M, D = s_mean_patients.shape
    L_, M_, D_ = s_mean_global.shape
    
    N_a = min(N_a, N_i)
    assert L == L_ and M == M_ and D == D_, f"shape mismatch: {s_mean_patients.shape} vs {s_mean_global.shape}"
    s_anatomy = s_mean_patients - s_mean_global[:, np.newaxis, :, :]
    s_anatomy = einops.rearrange(s_anatomy, "L N_i M D -> L N_i (M D)")
    P_anatomy_list = []
    b_anatomy_list = []
    lambda_list = []
    for s_anatomy_i in s_anatomy:
        pca = PCA(n_components=N_a)
        pca.fit(s_anatomy_i.astype(np.float64))
        P_anatomy_list.append(pca.components_)  # shape = (N_a, M*D)
        b_anatomy_list.append(pca.transform(s_anatomy_i))   # shape = (N_i, N_a)
        lambda_list.append(pca.explained_variance_)  # shape = (N_a,)
    P_anatomy = np.stack(P_anatomy_list)    # shape = (L, N_a, M*D)
    b_anatomy = np.stack(b_anatomy_list)    # shape = (L, N_i, N_a)
    lambda_ = np.stack(lambda_list) # shape = (L, N_a)
    P_anatomy = einops.rearrange(P_anatomy, "L N_a (M D) -> L N_a M D", M=M, D=D)
    return P_anatomy.astype(np.float32), b_anatomy.astype(np.float32), lambda_.astype(np.float32)

def calculate_motion_pca(ssm_points: np.ndarray, s_mean_patients: np.ndarray, N_m: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        ssm_points (np.ndarray): The SSM points array, shape = (num_labels(L), num_patients(N_i), num_phases(N_j), num_points(M), 3).
        s_mean_patients (np.ndarray): The mean patient points array, shape = (num_labels(L), num_patients(N_i), num_points(M), 3).
        N_m (int, optional): Number of motion components. Defaults to 7.
    Returns:
        Principal component direction, shape = (num_labels(L), num_components(N_m), num_points(M), 3)
        Principal component weights, shape = (num_labels(L), num_patients(N_i), num_phases(N_j), num_components(N_m))
        Explained variance(lambda), shape = (num_labels(L), num_components(N_m))
    """
    assert ssm_points.ndim == 5, f"ssm_points.ndim is {ssm_points.ndim}, expected 5"
    assert s_mean_patients.ndim == 4, f"s_mean_points.ndim is {s_mean_patients.ndim}, expected 4"
    
    L, N_i, N_j, M, D = ssm_points.shape
    L_, N_i_, M_, D_ = s_mean_patients.shape
    assert L == L_ and N_i == N_i_ and M == M_ and D == D_, f"shape mismatch: {ssm_points.shape} vs {s_mean_patients.shape}"
    s_motion = ssm_points - s_mean_patients[:, :, np.newaxis, :, :]
    s_motion = einops.rearrange(s_motion, "L N_i N_j M D -> L (N_i N_j) (M D)")
    
    P_motion_list = []
    b_motion_list = []
    lambda_list = []
    # 注：此处必须同时对多个相位的数据进行PCA，如此才能提取整个心动周期的运动趋势（运动主成分）。
    # 特征点（landmark）在整个行动周期的运动方向都是相同的————即P_motion中指定label, 主成分
    # 和特征点后所得的向量就是主要运动方法：v = P[i, j, k] = (x, y, z)
    # 如果对每个label、每个相位都计算PCA, 则不同患者的的位置差异性会掩盖运动趋势
    # 但这也存在问题，即每个特征点运动方向相同，可能导致运动较为僵硬
    for s_motion_i in s_motion:
        pca = PCA(n_components=N_m)
        pca.fit(s_motion_i.astype(np.float64))
        P_motion = pca.components_  # shape = (N_m, M*D)
        b_motion = pca.transform(s_motion_i)     # shape = (N_i * N_j, N_m)
        P_motion_list.append(P_motion)
        b_motion_list.append(b_motion)
        lambda_list.append(pca.explained_variance_)
    P_motion = np.stack(P_motion_list)
    b_motion = np.stack(b_motion_list)
    lambda_ = np.stack(lambda_list) # shape = (L, N_m)
    P_motion = einops.rearrange(P_motion, "L N_m (M D) -> L N_m M D", M=M, D=D)
    b_motion = einops.rearrange(b_motion, "L (N_i N_j) N_m -> L N_i N_j N_m", N_i=N_i, N_j=N_j)
    return P_motion.astype(np.float32), b_motion.astype(np.float32), lambda_.astype(np.float32)

def visualize_motion_deformation(
    P_motion: np.ndarray,  # (L, N_m, M, 3)
    b_motion_mean_per_phase: np.ndarray,  # (L, N_j, N_m)
    s_mean_global: np.ndarray,  # (L, M, 3)
    output_path: Path,
    num_components_used: None | int = None,
    gif_name: str = "motion_deformation.gif"
):
    num_labels, num_phases, num_components = b_motion_mean_per_phase.shape
    _, _, num_points, _ = P_motion.shape

    assert s_mean_global.shape == (num_labels, num_points, 3)
    
    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif(str(output_path / gif_name))

    # Create an initial full point cloud and label array
    full_points = []
    full_labels = []

    for label_id in range(num_labels):
        full_points.append(s_mean_global[label_id])
        full_labels.append(np.full((num_points,), label_id))

    full_points = np.vstack(full_points)  # shape: (L*M, 3)
    full_labels = np.concatenate(full_labels)  # shape: (L*M,)

    # Create pyvista point cloud mesh
    point_cloud = pv.PolyData(full_points)
    point_cloud["label"] = full_labels

    # Add the mesh
    plotter.add_mesh(point_cloud, scalars="label", render_points_as_spheres=True, point_size=5.0, show_scalar_bar=False)

    # Animate deformation per phase
    for phase_id in range(num_phases):
        deformed_points = []

        for label_id in range(num_labels):
            b = b_motion_mean_per_phase[label_id, phase_id]  # (N_m,)
            P = P_motion[label_id]  # (N_m, M, 3)
            
            if num_components_used is not None:
                b = b[:num_components_used]
                P = P[:num_components_used]
            
            deformation = np.tensordot(b, P, axes=([0], [0]))  # (M, 3)
            new_points = s_mean_global[label_id] + deformation  # (M, 3)
            deformed_points.append(new_points)

        new_full_points = np.vstack(deformed_points)
        point_cloud.points = new_full_points
        plotter.write_frame()

    plotter.close()
    print(f"Saved motion animation to {output_path / gif_name}")

@app.command()
def calculate_ssm(
    aligned_vtk_dir: Annotated[
        Path, typer.Argument(help="The directory containing the aligned VTK files.")
    ] = project_root / "data" / "output_ssm_vtk_aligned",
    output_dir: Annotated[
        Path, typer.Argument(help="The output directory for the Statistical Shape Model (SSM) results.")
    ] = project_root / "data" / "output_ssm_pca",
    template_surface: Annotated[
        Path, typer.Option(help="The template surface for the SSM, used to generate the new average surface template")
    ] = project_root / "ssm_template.vtk",
    visualize: Annotated[
        bool, typer.Option(help="Whether to visualize the motion deformation.")
    ] = False,
):
    """
    Calculate the Statistical Shape Model (SSM) for the aligned VTK files. [STEP 2]
    1) Calculate the mean points per patient, per phase, and globally.
    2) Calculate the anatomy PCA and motion PCA.
    """
    assert aligned_vtk_dir.is_dir(), f"{aligned_vtk_dir} is not a directory"
    assert template_surface.is_file(), f"{template_surface} is not a file"
    assert template_surface.suffix == ".vtk", f"{template_surface} is not a VTK file"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_vtk_points(aligned_vtk_dir)

    s_mean_patients = calculate_mean_points_per_patient(data)   # shape = (L, N_i, M, 3)
    s_mean_phases = calculate_mean_points_per_phase(data)   # shape = (L, N_j, M, 3)
    s_mean_global = calculate_mean_points(data)   # shape = (L, M, 3)

    # save s_mean as a new ssm_template
    mesh = pv.PolyData()
    for label_id in range(s_mean_global.shape[0]):
        points = s_mean_global[label_id]
        label = pv.PolyData(points)
        label.point_data["label"] = np.ones(points.shape[0]).astype(np.uint8) * (label_id+1)
        mesh = mesh.merge(label)
    template_surface = pv.read(template_surface)
    template_surface.points = mesh.points
    template_surface.save(output_dir / "ssm_template_avg.vtk")

    for phase_id in range(s_mean_phases.shape[1]):
        template_surface = template_surface.copy()
        template_surface.points = s_mean_phases[:, phase_id].reshape(-1, 3)
        p = output_dir / "avg_phases" / f"ssm_template_phase_{phase_id:03d}.vtk"
        p.parent.mkdir(parents=True, exist_ok=True)
        template_surface.save(p)

    # Calculate anatomy PCA
    P_anatomy, b_anatomy, lambda_anatomy = calculate_anatomy_pca(s_mean_patients, s_mean_global)
    # Calculate motion PCA
    P_motion, b_motion, lambda_motion = calculate_motion_pca(data, s_mean_patients)
    # Save results
    b_motion_mean_per_phase = np.mean(b_motion, axis=1)     # shape = (L, N_j, N_m)
    print(f"{P_anatomy.shape=}\n{b_anatomy.shape=}\n{lambda_anatomy.shape=}")
    print(f"{P_motion.shape=}\n{b_motion.shape=}\n{b_motion_mean_per_phase.shape=}\n{lambda_motion.shape=}")
    np.save(output_dir / "P_anatomy.npy", P_anatomy)
    np.save(output_dir / "b_anatomy.npy", b_anatomy)
    np.save(output_dir / "lambda_anatomy.npy", lambda_anatomy)
    np.save(output_dir / "P_motion.npy", P_motion)
    np.save(output_dir / "b_motion.npy", b_motion)
    np.save(output_dir / "b_motion_mean_per_phase.npy", b_motion_mean_per_phase)
    np.save(output_dir / "lambda_motion.npy", lambda_motion)
    np.save(output_dir / "s_mean_patients.npy", s_mean_patients)
    np.save(output_dir / "s_mean_global.npy", s_mean_global)
    print(f"Saved PCA results to {output_dir}")
    
    # Visualize motion deformation
    if visualize:
        for num_components_used in [1, 2, 3, 4, 5, 6, 7]:
            visualize_motion_deformation(
                P_motion=P_motion,
                b_motion_mean_per_phase=b_motion_mean_per_phase,
                s_mean_global=s_mean_global,
                output_path=output_dir,
                num_components_used=num_components_used,
                gif_name=f"motion_deformation_{num_components_used}.gif"
            )


if __name__ == "__main__":
    typer.run(calculate_ssm)
