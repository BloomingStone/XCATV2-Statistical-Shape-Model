#!/bin/bash

python script/generate_ssm_data.py --out_frames 20 &&
python script/generate_ssm_data.py --out_frames 20 --include_vessels --output_nii_name output_ssm_nii__with_vessels &&

python script/get_surface_cloud.py &&
python script/get_volume_cloud.py &&
python script/align_surface.py &&
python script/calculate_ssm_pca.py &&
python script/generate_4d_image.py