import io
import os
import json
import shutil
import sys

import SimpleITK as sitk

import configparser
from utils.Data_utils import parse_file_tree, copy
from utils.format_convert import dcm2nii, png2nii

file_root_path = '/mnt/ssd1/Datasets/MRI/ACDC/training'
file_tree = parse_file_tree(file_root_path)

target_path = '/mnt/ssd1/Datasets/MRI/ACDC/organized/Train'
os.makedirs(target_path, exist_ok=True)
with open(os.path.join(target_path, 'file_structure.json'), 'w') as f:
    json.dump(file_tree, f, indent=4)

# create organized data storage path
if not os.path.exists(os.path.join(target_path, 'images')):
    os.makedirs(os.path.join(target_path, 'images'))

if not os.path.exists(os.path.join(target_path, 'labels')):
    os.makedirs(os.path.join(target_path, 'labels'))


for subject_name, subject in file_tree.items():
    if isinstance(subject, dict):
        nii_files = [f for f in os.listdir(os.path.join(file_root_path, subject_name)) if
                     f.endswith('.nii') or f.endswith('.nii.gz')]
        for nii_file in nii_files:
            if '4d' in nii_file:
                continue
            if 'gt' in nii_file:
                gt_file_path = os.path.join(file_root_path, subject_name, nii_file)
                target_gt_file_path = os.path.join(target_path, 'labels', nii_file.replace('_gt', ''))
                copy(gt_file_path, target_gt_file_path)
            else:
                image_file_path = os.path.join(file_root_path, subject_name, nii_file)
                target_image_file_path = os.path.join(target_path, 'images', nii_file)
                copy(image_file_path, target_image_file_path)

