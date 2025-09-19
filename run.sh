#!/bin/bash

python script/_01_generate_ssm_data.py --out_frames 20 &&
# python script/_01_generate_ssm_data.py --out_frames 20 --include_vessels --output_nii_name output_ssm_nii__with_vessels &&

python script/_02_get_surface_cloud.py &&
python script/_03_get_volume_cloud.py &&
python script/_04_align_surface.py &&
python script/_05_calculate_ssm_pca.py &&
python script/_06_generate_4d_image.py