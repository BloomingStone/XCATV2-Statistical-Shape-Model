# %%
from pathlib import Path
import multiprocessing
from functools import partial
import os
import logging
from tqdm import tqdm

import pyvista as pv
pv.set_jupyter_backend('html')  # 恢复jupyter后端设置
import pyacvd
import nibabel as nib
import numpy as np


ssm_nii_dir = Path.cwd() / "output_ssm_nii"
vtk_dir = Path.cwd() / "output_ssm_vtk_volume"

# %%
# 配置日志记录
logging.basicConfig(
    filename='processing_errors.log',
    level=logging.ERROR,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    encoding='utf-8'
)

def process_label_file(label_nii_path: Path, vtk_dir: Path):
    """处理单个label文件并保存VTK结果"""
    max_point_num = 10000
    try:
        ssm_case_name = label_nii_path.parent.parent.name
        label_name = label_nii_path.stem.split(".")[0]
        vtk_file = vtk_dir / ssm_case_name / f"{label_name}.vtk"
        vtk_file.parent.mkdir(parents=True, exist_ok=True)
        
        label_nii = nib.load(str(label_nii_path))
        label_data = label_nii.get_fdata()
        label_ids = sorted(np.unique(label_data).astype(np.int8))[1:]
        volume_points_all = pv.PolyData()
        
        for label_id in label_ids:
            assert label_id > 0
            points = np.argwhere(label_data == label_id)
            stride = max(1, len(points) // max_point_num)
            points = points[::stride]
            volume_points = pv.PolyData(points)
            volume_points.point_data["label"] = np.ones(volume_points.n_points).astype(np.uint8) * label_id
            volume_points_all = volume_points_all.merge(volume_points)
        
        volume_points_all.save(str(vtk_file))
        return True
    except Exception as e:
        error_msg = f"Failed to process {label_nii_path}"
        logging.error(error_msg, exc_info=True)  # 记录完整错误堆栈
        print(f"ERROR: {error_msg} - see processing_errors.log for details")
        return False

# 收集所有需要处理的label文件路径
label_files = []
for ssm_case in ssm_nii_dir.iterdir():
    if ssm_case.is_dir():
        label_dir = ssm_case / "label"
        label_files.extend(sorted(label_dir.glob("*.nii.gz")))

# 设置进程池 (保留1个核心给系统)
num_workers = max(1, os.cpu_count() - 1)
        
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
