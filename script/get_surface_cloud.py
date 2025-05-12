from pathlib import Path
import multiprocessing
from functools import partial
import os
import logging
from tqdm import tqdm

import pyvista as pv
import pyacvd
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from scipy import ndimage

def get_largest_connected_component(data: np.ndarray) -> np.ndarray:
    """获取二值图像中最大的连通区域"""
    structure = ndimage.generate_binary_structure(3, 1)
    labeled_data, num_features = ndimage.label(data, structure=structure)
    sizes = ndimage.sum(data, labeled_data, range(num_features + 1))
    largest_component = sizes.argmax()
    return (labeled_data == largest_component).astype(np.uint8)

# 配置日志记录
logging.basicConfig(
    filename='processing_errors.log',
    level=logging.ERROR,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    encoding='utf-8'
)

# TODO 可能需要更好地处理这里的direction
# TODO 可能需要进行缩放
def get_cloud_from_nii_label(label_nii_path: Path, max_point_num, direction = (-1, 1, 1)):
    """从NII文件中提取点云"""
    label_nii = nib.load(str(label_nii_path))
    label_data = label_nii.get_fdata()
    direction_label = np.diag(label_nii.affine[:3, :3])
    for i in range(3):
        if direction[i] * direction_label[i] < 0:
            label_data = np.flip(label_data, axis=i)
    
    label_ids = sorted(np.unique(label_data).astype(np.int8))[1:]
    surface_all = pv.PolyData()
    
    for label_id in label_ids:
        assert label_id > 0
        if label_id == 1:
            # 左室心肌部分，和左室腔部分合并
            mask_1 = (label_data == 1).astype(np.uint8)
            mask_2 = (label_data == 2).astype(np.uint8)
            mask = mask_1 + mask_2
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = (label_data == label_id).astype(np.uint8)
        
        structure = ndimage.generate_binary_structure(3, 1)
        mask = get_largest_connected_component(mask)
        mask = binary_closing(mask, iterations=1, structure=structure)
        mask = binary_opening(mask, iterations=1, structure=structure)

        surface = pv.wrap(mask).contour([1], method="flying_edges").triangulate().smooth_taubin(n_iter=50).clean()
        cluster = pyacvd.Clustering(surface)
        cluster.subdivide(2)
        cluster.cluster(max_point_num)
        surface = cluster.create_mesh().triangulate().clean()
        if np.isnan(surface.points).any():
            raise ValueError(f"NaN in points")
        if not surface.is_manifold:
            raise ValueError(f"Mesh is not manifold")
        surface.point_data["label"] = np.ones(surface.n_points).astype(np.uint8) * label_id
        surface_all = surface_all.merge(surface)
    return surface_all

def process_label_file(label_nii_path: Path, vtk_dir: Path, max_point_num = 2000):
    """处理单个label文件并保存VTK结果"""
    try:
        ssm_case_name = label_nii_path.parent.parent.name
        label_name = label_nii_path.stem.split(".")[0]
        vtk_file = vtk_dir / ssm_case_name / f"{label_name}.vtk"
        vtk_file.parent.mkdir(parents=True, exist_ok=True)
        
        surface_all = get_cloud_from_nii_label(label_nii_path, max_point_num)
        surface_all.save(str(vtk_file))
        return True
    except Exception as e:
        error_msg = f"Failed to process {label_nii_path}"
        logging.error(error_msg, exc_info=True)  # 记录完整错误堆栈
        print(f"ERROR: {error_msg} - see processing_errors.log for details")
        return False

def main_multi_process(ssm_nii_dir: Path, vtk_dir: Path, max_point_num = 2000):
    # 收集所有需要处理的label文件路径
    label_files = []
    for ssm_case in ssm_nii_dir.iterdir():
        if ssm_case.is_dir():
            label_dir = ssm_case / "label"
            label_files.extend(sorted(label_dir.glob("*.nii.gz")))

    # 设置进程池 (保留1个核心给系统)
    num_workers = max(1, os.cpu_count()-1)
            
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 创建偏函数传递固定参数
        worker_func = partial(
            process_label_file, 
            vtk_dir=vtk_dir,
            max_point_num=max_point_num
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
    for ssm_case in tqdm(ssm_nii_dir.iterdir()):
        if not ssm_case.is_dir():
            continue
        label_dir = ssm_case / "label"
        for label_nii_path in tqdm(label_dir.glob("*.nii.gz")):
            process_label_file(
                label_nii_path=label_dir / label_nii_path.name,
                vtk_dir=vtk_dir
            )

if __name__ == "__main__":
    ssm_nii_dir = Path.cwd() / "output_ssm_nii"
    vtk_dir = Path.cwd() / "output_ssm_vtk"
    main_multi_process(ssm_nii_dir=ssm_nii_dir, vtk_dir=vtk_dir, max_point_num=200)
    # main_single_process(ssm_nii_dir=ssm_nii_dir, vtk_dir=vtk_dir)