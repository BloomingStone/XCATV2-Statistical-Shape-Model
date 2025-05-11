from pathlib import Path
import logging
import multiprocessing
import os

import torchcpd
import pyvista as pv
import torch
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass

logging.basicConfig(
    filename='aligning_errors.log',
    level=logging.ERROR,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    encoding='utf-8'
)

def align_surface_rigid(
        moving_surface: pv.PolyData, 
        moving_point_cloud: pv.PolyData, 
        fixed_point_cloud: pv.PolyData,
        device: str = 'cuda:0'
    ) -> pv.PolyData:
    """Align two surfaces using rigid transformation"""
    source_points = moving_point_cloud.points
    target_points = fixed_point_cloud.points
    res = moving_surface.copy()
    reg = torchcpd.RigidRegistration(X=target_points[::10], Y=source_points[::10], device=device, scale=False)
    _, (s, R, t) = reg.register()
    # handel nan in translation:
    if torch.isnan(t).any():
        raise ValueError(f"NaN in translation")

    res.points = reg.transform_point_cloud(torch.tensor(moving_surface.points, device=device, dtype=torch.float64)).cpu().numpy()
    return res

def deform_surface(
        source_surface: pv.PolyData, 
        target_surface: pv.PolyData, 
        device: str = 'cuda:0',
        **deform_kwargs
    ) -> pv.PolyData:
    moving_labels = source_surface.point_data["label"]
    fix_labels = target_surface.point_data["label"]
    labels = np.unique(moving_labels)
    labels = labels[labels != 0]
    new_cloud = pv.PolyData()
    for label in labels:
        source_points = source_surface.points[moving_labels == label]
        target_points = target_surface.points[fix_labels == label]
        new_points, _ = torchcpd.RigidRegistration(X=target_points, Y=source_points, device=device).register()
        new_points, _ = torchcpd.AffineRegistration(X=target_points, Y=new_points.cpu().numpy(), device=device).register()
        new_points, _ = torchcpd.DeformableRegistration(X=target_points, Y=new_points.cpu().numpy(), device=device, kwargs=deform_kwargs).register()
        cloud = pv.PolyData(new_points.cpu().numpy())
        cloud.point_data["label"] = np.ones(cloud.n_points).astype(np.uint8) * label
        new_cloud = new_cloud.merge(cloud)
    
    res = source_surface.copy()
    res.points = new_cloud.points
    return res


@dataclass
class ProcessCaseArgs:
    mov_vtk_file: Path
    mov_volume_point_cloud_file: Path
    vtk_dir: Path
    aligned_vtk_dir: Path
    fix_volume_point_cloud: pv.PolyData
    template_surface: pv.PolyData
    gpu_id: int

def process_case(args: ProcessCaseArgs) -> Path | None:
    """
    Align surface and deform surface to template surface
    step 1: align surface by the transformation of the volume point cloud (moving volume point cloud -> fixed volume point cloud)
    step 2: deform template surface to the aligned surface for generate
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = 'cuda'
    
    try:
        mov_surface = pv.read(args.mov_vtk_file)
        mov_volume_point_cloud = pv.read(args.mov_volume_point_cloud_file)
        
        mov_surface = align_surface_rigid(
            moving_surface=mov_surface,
            moving_point_cloud=mov_volume_point_cloud,
            fixed_point_cloud=args.fix_volume_point_cloud,
            device=device
        )

        mov_surface = deform_surface(
            source_surface=args.template_surface,
            target_surface=mov_surface,
            device=device
        )

        new_vtk_file = args.aligned_vtk_dir / args.mov_vtk_file.relative_to(args.vtk_dir)
        new_vtk_file.parent.mkdir(parents=True, exist_ok=True)
        mov_surface.save(new_vtk_file)
        return None
    except Exception as e:
        logging.error(f"Failed to process {args.mov_vtk_file}", exc_info=True)
        return args.mov_vtk_file
    finally:
        torch.cuda.empty_cache()

def main(template_surface_path: Path | None = None):
    vtk_dir = Path.cwd() / "output_ssm_vtk"
    aligned_vtk_dir = Path.cwd() / "output_ssm_vtk_aligned"
    volume_points_cloud_dir = Path.cwd() / "output_ssm_vtk_volume"
    surface_vtk_files: dict[str: list[Path]] = {
        case_dir.name: sorted(case_dir.glob("*.vtk")) for case_dir in vtk_dir.iterdir() if case_dir.is_dir()
    }

    fix_case_name = sorted(surface_vtk_files.keys())[0]
    fix_case_files = surface_vtk_files[fix_case_name].copy()
    del surface_vtk_files[fix_case_name]
    print(f'fixed case: {fix_case_name}')

    if template_surface_path is None:
        template_surface = pv.read(fix_case_files[0])
        template_surface.save('ssm_template.vtk')
    else:
        template_surface = pv.read(template_surface_path)

    # Prepare fixed data for all phases first
    fix_volume_point_cloud_list = []
    for phase in range(10):
        fix_vtk_file = fix_case_files[phase]
        fix_surface = pv.read(fix_vtk_file)
        new_fix_vtk_file = aligned_vtk_dir / fix_vtk_file.relative_to(vtk_dir)
        new_fix_vtk_file.parent.mkdir(parents=True, exist_ok=True)
        fix_surface.save(new_fix_vtk_file)

        fix_volume_point_cloud = pv.read(volume_points_cloud_dir / fix_vtk_file.relative_to(vtk_dir))
        fix_volume_point_cloud_list.append(fix_volume_point_cloud)
    

    failed_cases = []
    num_gpus = 4
    pool = multiprocessing.Pool(processes=num_gpus)

    with multiprocessing.Pool(processes=num_gpus) as pool:
        for phase in tqdm(range(10), desc="Processing phases"):
            task_args = []
            for index, case_files in enumerate(surface_vtk_files.values()):
                gpu_id = index % num_gpus  # Distribute cases evenly across GPUs
                task_args.append(ProcessCaseArgs(
                    mov_vtk_file=case_files[phase],
                    mov_volume_point_cloud_file=volume_points_cloud_dir / case_files[phase].relative_to(vtk_dir),
                    vtk_dir=vtk_dir,
                    aligned_vtk_dir=aligned_vtk_dir,
                    fix_volume_point_cloud=fix_volume_point_cloud_list[phase].copy(),
                    template_surface=template_surface.copy(),
                    gpu_id=gpu_id
                ))

            # Process cases in parallel
            with tqdm(total=len(task_args), desc="Processing files") as pbar:
                results = pool.imap_unordered(process_case, task_args)
                for result in results:
                    pbar.update(1)
                    if result:
                        failed_cases.append(result)
                        print(f"ERROR: Failed to process {result} - see aligning_errors.log for details")

    print(f"Failed cases: {failed_cases}")

if __name__ == "__main__":
    main(template_surface_path=Path("/media/data3/sj/Data/Phatom/output_ssm_pca/ssm_template_avg.vtk"))
    # main()
