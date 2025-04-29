from pathlib import Path
import multiprocessing
from functools import partial
import os
import logging
from tqdm import tqdm

import pyvista as pv
import nibabel as nib
import numpy as np


# 配置日志记录
logging.basicConfig(
    filename='processing_errors.log',
    level=logging.ERROR,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    encoding='utf-8'
)

def process_label_file(label_nii_path: Path, vtk_dir: Path, max_point_num = 1000):
    """处理单个label文件并保存VTK结果"""
    try:
        ssm_case_name = label_nii_path.parent.parent.name
        label_name = label_nii_path.stem.split(".")[0]
        vtk_file = vtk_dir / ssm_case_name / f"{label_name}.vtk"
        vtk_file.parent.mkdir(parents=True, exist_ok=True)
        
        label_nii = nib.load(str(label_nii_path))
        label_data = label_nii.get_fdata()
        label_ids = sorted(np.unique(label_data).astype(np.int8))[1:]
        surface_all = pv.PolyData()
        
        for label_id in label_ids:
            assert label_id > 0
            surface = pv.wrap((label_data == label_id).astype(np.uint8)).contour([1], method="flying_edges").triangulate().smooth_taubin()
            if np.isnan(surface.points).any():
                raise ValueError(f"NaN in points, vtk_file: {vtk_file}")
            if surface.n_points > max_point_num:
                surface = pv.PolyData(surface.points[np.random.choice(surface.n_points, max_point_num, replace=False)])
            else:
                surface = pv.PolyData(surface.points)
            surface.point_data["label"] = np.ones(surface.n_points).astype(np.uint8) * label_id
            surface_all = surface_all.merge(surface)
        
        surface_all.save(str(vtk_file))
        return True
    except Exception as e:
        error_msg = f"Failed to process {label_nii_path}"
        logging.error(error_msg, exc_info=True)  # 记录完整错误堆栈
        print(f"ERROR: {error_msg} - see processing_errors.log for details")
        return False

def main_multi_process(ssm_nii_dir: Path, vtk_dir: Path):
    # 收集所有需要处理的label文件路径
    label_files = []
    for ssm_case in ssm_nii_dir.iterdir():
        if ssm_case.is_dir():
            label_dir = ssm_case / "label"
            label_files.extend(sorted(label_dir.glob("*.nii.gz")))

    # 设置进程池 (保留1个核心给系统)
    num_workers = max(1, os.cpu_count()//2)
            
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 创建偏函数传递固定参数
        worker_func = partial(
            process_label_file, 
            vtk_dir=vtk_dir
        )
        
        # 使用tqdm显示美观进度条
        with tqdm(total=len(label_files), desc="Processing files") as pbar:
            results = []
            for result in pool.imap_unordered(worker_func, label_files):
                results.append(result)
                pbar.update(1)
                
        # 输出统计信息
        success_count = sum(results)
        print(f"\nProcessing complete! Success: {success_count}/{len(label_files)}")

def main_single_process(ssm_nii_dir: Path, vtk_dir: Path):
    # 收集所有需要处理的label文件路径
    for ssm_case in ssm_nii_dir.iterdir():
        if not ssm_case.is_dir():
            continue
        label_dir = ssm_case / "label"
        for label_nii_path in label_dir.glob("*.nii.gz"):
            process_label_file(
                label_nii_path=label_dir / label_nii_path.name,
                vtk_dir=vtk_dir
            )

if __name__ == "__main__":
    ssm_nii_dir = Path.cwd() / "output_ssm_nii"
    vtk_dir = Path.cwd() / "output_ssm_vtk"
    main_multi_process(ssm_nii_dir=ssm_nii_dir, vtk_dir=vtk_dir)