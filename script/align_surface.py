from pathlib import Path
import logging

import torchcpd
import pyvista as pv
import torch
from tqdm import tqdm
import numpy as np

logging.basicConfig(
    filename='aligning_errors.log',
    level=logging.ERROR,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    encoding='utf-8'
)

def align_surface_rigid(moving_surface: pv.PolyData, moving_point_cloud: pv.PolyData, fixed_point_cloud: pv.PolyData) -> pv.PolyData:
    """Align two surfaces using rigid transformation"""
    source_points = moving_point_cloud.points
    target_points = fixed_point_cloud.points
    res = moving_surface.copy()
    reg = torchcpd.RigidRegistration(X=target_points[::10], Y=source_points[::10], device='cuda', scale=False)
    _, (s, R, t) = reg.register()
    # handel nan in translation:
    if torch.isnan(t).any():
        raise ValueError(f"NaN in translation")

    res.points = reg.transform_point_cloud(torch.tensor(moving_surface.points, device='cuda', dtype=torch.float64)).cpu().numpy()
    return res

def deform_surface(source_surface: pv.PolyData, target_surface: pv.PolyData, **deform_kwargs) -> pv.PolyData:
    moving_labels = source_surface.point_data["label"]
    fix_labels = target_surface.point_data["label"]
    res = pv.PolyData()
    labels = np.unique(moving_labels)
    labels = labels[labels != 0]
    for label in labels:
        source_points = source_surface.points[moving_labels == label]
        target_points = target_surface.points[fix_labels == label]
        new_points, _ = torchcpd.RigidRegistration(X=target_points, Y=source_points, device='cuda').register()
        new_points, _ = torchcpd.AffineRegistration(X=target_points, Y=new_points.cpu().numpy(), device='cuda').register()
        new_points, _ = torchcpd.DeformableRegistration(X=target_points, Y=new_points.cpu().numpy(), device='cuda', kwargs=deform_kwargs).register()
        cloud = pv.PolyData(new_points.cpu().numpy())
        cloud.point_data["label"] = np.ones(cloud.n_points).astype(np.uint8) * label
        res = res.merge(cloud)
    
    return res

def main():
    vtk_dir = Path.cwd() / "output_ssm_vtk"
    aligned_vtk_dir = Path.cwd() / "output_ssm_vtk_aligned"
    volume_points_cloud_dir = Path.cwd() / "output_ssm_vtk_volume"
    surface_vtk_files: dict[str: list[Path]] = {
        case_dir.name: sorted(case_dir.glob("*.vtk")) for case_dir in vtk_dir.iterdir() if case_dir.is_dir()
    }
    print(len(surface_vtk_files))

    fix_case_name = sorted(surface_vtk_files.keys())[0]
    fix_case_files = surface_vtk_files[fix_case_name].copy()
    print(f"{fix_case_name=}")

    template_vtk_file = fix_case_files[0]
    template_surface = pv.read(template_vtk_file)
    template_surface.save('ssm_template.vtk')

    failed_cases = []
    for phase in tqdm(range(10), desc="Processing phases"):
        fix_vtk_file = fix_case_files[phase]
        fix_surface = pv.read(fix_vtk_file)
        new_fix_vtk_file = aligned_vtk_dir / fix_vtk_file.relative_to(vtk_dir)
        new_fix_vtk_file.parent.mkdir(parents=True, exist_ok=True)
        fix_surface.save(new_fix_vtk_file)
        fix_point_cloud = pv.read(volume_points_cloud_dir / fix_vtk_file.relative_to(vtk_dir))
        for case_name, case_files in tqdm(surface_vtk_files.items(), desc="Processing cases"):
            mov_vtk_file = case_files[phase]
            mov_surface = pv.read(mov_vtk_file)
            mov_point_cloud = pv.read(volume_points_cloud_dir / mov_vtk_file.relative_to(vtk_dir))
            try:
                mov_surface = align_surface_rigid(
                    moving_surface=mov_surface,
                    moving_point_cloud=mov_point_cloud,
                    fixed_point_cloud=fix_point_cloud
                )

                mov_surface = deform_surface(
                    source_surface=template_surface,
                    target_surface=mov_surface
                )
            except Exception as e:
                logging.error(f"Failed to process {mov_vtk_file}", exc_info=True)
                print(f"ERROR: Failed to process {mov_vtk_file} - see aligning_errors.log for details")
                failed_cases.append(mov_vtk_file)
                continue

            new_vtk_file = aligned_vtk_dir / mov_vtk_file.relative_to(vtk_dir)
            new_vtk_file.parent.mkdir(parents=True, exist_ok=True)
            mov_surface.save(new_vtk_file)

    print(f"Failed cases: {failed_cases}")

if __name__ == "__main__":
    main()