import subprocess
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from multiprocessing import Pool
from datetime import datetime
import argparse
import logging
import os

import numpy as np
import nibabel as nib

from resolve_par import resolve_par


logging.basicConfig(
    filename='processing_errors.log',
    level=logging.ERROR,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    encoding='utf-8'
)

def read_numpy_from_bin(raw_bin_file: Path, output_shape: tuple[int, int, int], *, is_label: bool) -> np.ndarray:
    with open(raw_bin_file, "rb") as f:
        raw_data = np.fromfile(f, dtype=np.float32)
    x, y, z = output_shape
    expected_shape = (z, y, x) # 原始轴顺序为(z, y, x)
    expected_size = np.prod(expected_shape)
    assert raw_data.size == expected_size
    data_3d = raw_data.reshape(expected_shape) # 原始轴顺序为(z, y, x)
    data_3d = data_3d.transpose(2, 1, 0) # 转换为(x, y, z)
    data_3d = np.flip(data_3d, axis=1)   # 原始图像的y轴向下，转换为向上 (对应AP方向)
    if is_label:
        data_3d = data_3d.astype(np.uint8)
    return data_3d

def cut_roi(image_data: np.ndarray, label_data: np.ndarray, roi_shape: tuple[int, int, int]=(144, 144, 128)) -> tuple[np.ndarray, np.ndarray]:
    mask = label_data > 0
    if not np.any(mask):
        raise ValueError("No positive labels found")
    
    shape = image_data.shape
    for i in range(3):
        if roi_shape[i] > shape[i]:
            raise ValueError("The shape of ROI bigger than the shape of image data")
    
    coords = np.where(mask)
    
    def get_min_max(i: int):
        min_i, max_i = np.min(coords[i]), np.max(coords[i])
        center_i = (min_i + max_i) // 2
        min_i = max(0, center_i - roi_shape[i] // 2)
        min_i = min(min_i, shape[i] - roi_shape[i])
        max_i = min_i + roi_shape[i]
        return min_i, max_i
    
    roi_box = [get_min_max(k) for k in range(3)]
    
    image_cropped = image_data[
        roi_box[0][0]:roi_box[0][1],
        roi_box[1][0]:roi_box[1][1], 
        roi_box[2][0]:roi_box[2][1]
    ]
    
    label_cropped = label_data[
        roi_box[0][0]:roi_box[0][1],
        roi_box[1][0]:roi_box[1][1], 
        roi_box[2][0]:roi_box[2][1]
    ]
    return image_cropped, label_cropped

def generate_ssm_file(
        temp_dir: Path,
        param_file: Path,
        nrb_file: Path,
        heart_nrb_file: Path,
        xcat_dir: Path,
        output_nii_dir: Path,
        output_prefix: str,
        gender: str,
        start_slice: int,
        end_slice: int,
        out_frames: int, 
        array_size: int,
        depth: int,
        spacing: tuple[int, int, int],
        include_vessels: bool
) -> None:
    print(f"{nrb_file.stem} start")
    (nii_image_dir := output_nii_dir / "image").mkdir(parents=True, exist_ok=True)
    (nii_label_dir := output_nii_dir / "label").mkdir(parents=True, exist_ok=True)
    
    with TemporaryDirectory(dir=temp_dir) as raw_dir_name:
        raw_dir = Path(raw_dir_name)
        args = [
            "./dxcat2",
            str(param_file),
            "--organ_file", str(nrb_file),
            "--heart_base", str(heart_nrb_file),
            "--gender",     "0" if gender == "male" else "1",
            "--startslice", str(start_slice),
            "--endslice", str(end_slice),
            "--out_frames",  str(out_frames),
        ]

        if include_vessels:
            args = args + [
                "--vessel_flag", "1",
                "--coronary_art_flag",  "1",
                "--coronary_vein_flag", "1",
                "--papillary_flag", "1",
                "--coronary_art_activity", "6",
                "--coronary_vein_activity", "7"
            ]
        
        args.append(str(raw_dir/output_prefix))
        
        subprocess.run(args, cwd=str(xcat_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for i in range(out_frames):
            raw_image_file = raw_dir / f"{output_prefix}_atn_{i + 1}.bin"
            raw_label_file = raw_dir / f"{output_prefix}_act_{i + 1}.bin"
            
            output_shape = (array_size, array_size, depth)
            image_data, label_data = cut_roi(
                read_numpy_from_bin(raw_image_file, output_shape, is_label=False), 
                read_numpy_from_bin(raw_label_file, output_shape, is_label=True)
            )
            
            affine = np.eye(4)
            affine[(0,1,2), (0,1,2)] = spacing
            nib.save(nib.nifti1.Nifti1Image(image_data, affine), nii_image_dir / f'{output_prefix}_{i:03d}.nii.gz')
            nib.save(nib.nifti1.Nifti1Image(label_data, affine), nii_label_dir / f'{output_prefix}_{i:03d}.nii.gz')
    print(f"{nrb_file.stem} done")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_frames", type=int, default=10)
    parser.add_argument("--include_vessels", action="store_true")
    parser.add_argument("--output_nii_name", type=str, default="output_ssm_nii")
    args = parser.parse_args()

    phantom_root_dir = Path.cwd().resolve()
    xcat_dir = phantom_root_dir / "XCAT"
    output_nii_root_dir = phantom_root_dir / args.output_nii_name
    source_nrb_dir = phantom_root_dir / "xcat_adult_nrb_files"
    (temp_dir := phantom_root_dir / "temp").mkdir(exist_ok=True)

    output_prefix = "ssm"

    param_file = phantom_root_dir / "parameters" / "statistical_shape_model_base.par"
    # param_file = phantom_root_dir / "parameters" / "statistical_shape_model_base__with_coronary_and_papillary.par"
    params = resolve_par(param_file)
    pixel_width_cm = params["pixel_width"]
    slice_width_cm = params["slice_width"]
    array_size = int(params["array_size"])
    start_slice_dict = {
        "male": int(110 / slice_width_cm),
        "female": int(103 / slice_width_cm),
    }
    end_slice_dict = {
        "male": int(155 / slice_width_cm),
        "female": int(148 / slice_width_cm)
    }
    depth_dict = {gender: int(end_slice_dict[gender] - start_slice_dict[gender] + 1) for gender in ["male", "female"]}
    
    out_frames = args.out_frames
    include_vessels = args.include_vessels
    pixel_width_mm = pixel_width_cm * 10
    slice_width_mm = slice_width_cm * 10
    spacing = (-pixel_width_mm, pixel_width_mm, slice_width_mm)

    worker_args = []
    for nrb_file in sorted(source_nrb_dir.glob("*.nrb"), reverse=True):
        match = re.search(r'(male|female)_pt(\d{1,3}).nrb', str(nrb_file))
        if not match:
            continue
        gender = match.group(1)
        number = match.group(2)
        
        case_name = f"{gender}_pt{number}"
        heart_nrb_file = nrb_file.with_name(f"{case_name}_heart.nrb")
        
        output_nii_dir = output_nii_root_dir / case_name
        start_slice = start_slice_dict[gender]
        end_slice = end_slice_dict[gender]
        depth = depth_dict[gender]

        worker_args.append((
            temp_dir,
            param_file,
            nrb_file,
            heart_nrb_file,
            xcat_dir,
            output_nii_dir,
            output_prefix,
            gender,
            start_slice,
            end_slice,
            out_frames,
            array_size,
            depth,
            spacing,
            include_vessels
        ))

    with Pool(processes=(os.cpu_count()//2)) as pool:
        pool.starmap(generate_ssm_file, worker_args)

if __name__ == "__main__":
    start = datetime.now()
    print(f"start: {start.strftime("%Y-%m-%d %H:%M:%S")}")
    main()
    end = datetime.now()
    print(f"end: {end.strftime("%Y-%m-%d %H:%M:%S")}")
    print(f"cost {(end-start)}")
