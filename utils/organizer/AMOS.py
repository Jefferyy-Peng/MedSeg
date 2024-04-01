import io
import os
import json
import shutil
import sys

import SimpleITK as sitk

import configparser
from utils.Data_utils import parse_file_tree, copy
from utils.format_convert import dcm2nii, png2nii

file_root_path = '/mnt/ssd1/Datasets/MRI/AMOS'
file_tree = parse_file_tree(file_root_path)

target_path = '/mnt/ssd1/Datasets/MRI/AMOS/organized'
os.makedirs(target_path, exist_ok=True)
with open(os.path.join(target_path, 'file_structure.json'), 'w') as f:
    json.dump(file_tree, f, indent=4)

# create organized data storage path
os.makedirs(os.path.join(target_path, 'Train', 'images'), exist_ok=True)
os.makedirs(os.path.join(target_path, 'Train', 'labels'), exist_ok=True)
os.makedirs(os.path.join(target_path, 'Test', 'images'), exist_ok=True)
os.makedirs(os.path.join(target_path, 'Test', 'labels'), exist_ok=True)


for subject_name, subject in file_tree.items():
    if isinstance(subject, dict):
        if subject_name == 'imagesTr' or subject_name == 'imagesVa':
            nii_files = [f for f in os.listdir(os.path.join(file_root_path, subject_name)) if
                         f.endswith('.nii') or f.endswith('.nii.gz')]
            for nii_file in nii_files:
                image_file_path = os.path.join(file_root_path, subject_name, nii_file)
                target_image_file_path = os.path.join(target_path, 'Train', 'images', nii_file)
                copy(image_file_path, target_image_file_path)
        elif subject_name == 'imagesTs':
            nii_files = [f for f in os.listdir(os.path.join(file_root_path, subject_name)) if
                         f.endswith('.nii') or f.endswith('.nii.gz')]
            for nii_file in nii_files:
                image_file_path = os.path.join(file_root_path, subject_name, nii_file)
                target_image_file_path = os.path.join(target_path, 'Test', 'images', nii_file)
                copy(image_file_path, target_image_file_path)
        elif subject_name == 'labelsTr' or subject_name == 'labelsVa':
            nii_files = [f for f in os.listdir(os.path.join(file_root_path, subject_name)) if
                         f.endswith('.nii') or f.endswith('.nii.gz')]
            for nii_file in nii_files:
                image_file_path = os.path.join(file_root_path, subject_name, nii_file)
                target_image_file_path = os.path.join(target_path, 'Train', 'labels', nii_file)
                copy(image_file_path, target_image_file_path)
        elif subject_name == 'labelsTs':
            nii_files = [f for f in os.listdir(os.path.join(file_root_path, subject_name)) if
                         f.endswith('.nii') or f.endswith('.nii.gz')]
            for nii_file in nii_files:
                image_file_path = os.path.join(file_root_path, subject_name, nii_file)
                target_image_file_path = os.path.join(target_path, 'Test', 'labels', nii_file)
                copy(image_file_path, target_image_file_path)

