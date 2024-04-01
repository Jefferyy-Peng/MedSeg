import os
import json

import SimpleITK as sitk
import numpy as np

from utils.Data_utils import parse_file_tree
from utils.format_convert import dcm2nii, png2nii

file_root_path = '/mnt/ssd1/Datasets/MRI/CHAO/Train_Sets/MR'
file_tree = parse_file_tree(file_root_path)

target_path = '/mnt/ssd1/Datasets/MRI/CHAO/organized/Train'
with open(os.path.join(target_path, 'file_structure.json'), 'w') as f:
    json.dump(file_tree, f, indent=4)

# create organized data storage path
if not os.path.exists(os.path.join(target_path, 'images')):
    os.makedirs(os.path.join(target_path, 'images'))

if not os.path.exists(os.path.join(target_path, 'labels')):
    os.makedirs(os.path.join(target_path, 'labels'))

for subject_name, subject in file_tree.items():
    dcm2nii(os.path.join(file_root_path, subject_name, 'T1DUAL', 'DICOM_anon', 'InPhase'), os.path.join(target_path, 'images', f'{subject_name}_T1DUAL_InPhase.nii.gz'))
    png2nii(os.path.join(file_root_path, subject_name, 'T1DUAL', 'Ground'), os.path.join(target_path, 'labels', f'{subject_name}_T1DUAL_InPhase.nii.gz'))
    gt_sitk = sitk.ReadImage(os.path.join(target_path, 'labels', f'{subject_name}_T1DUAL_InPhase.nii.gz'))
    gt_data = sitk.GetArrayFromImage(gt_sitk)
    gt_data[np.logical_and(gt_data >= 55, gt_data <= 70)] = 1
    gt_data[np.logical_and(gt_data >= 110, gt_data <= 135)] = 2
    gt_data[np.logical_and(gt_data >= 175, gt_data <= 200)] = 3
    gt_data[np.logical_and(gt_data >= 240, gt_data <= 255)] = 4
    sitk_image = sitk.GetImageFromArray(gt_data)
    sitk.WriteImage(sitk_image, os.path.join(target_path, 'labels', f'{subject_name}_T1DUAL_InPhase.nii.gz'))
    dcm2nii(os.path.join(file_root_path, subject_name, 'T1DUAL', 'DICOM_anon', 'OutPhase'),
            os.path.join(target_path, 'images', f'{subject_name}_T1DUAL_OutPhase.nii.gz'))
    png2nii(os.path.join(file_root_path, subject_name, 'T1DUAL', 'Ground'),
            os.path.join(target_path, 'labels', f'{subject_name}_T1DUAL_OutPhase.nii.gz'))
    gt_sitk = sitk.ReadImage(os.path.join(target_path, 'labels', f'{subject_name}_T1DUAL_OutPhase.nii.gz'))
    gt_data = sitk.GetArrayFromImage(gt_sitk)
    gt_data[np.logical_and(gt_data >= 55, gt_data <= 70)] = 1
    gt_data[np.logical_and(gt_data >= 110, gt_data <= 135)] = 2
    gt_data[np.logical_and(gt_data >= 175, gt_data <= 200)] = 3
    gt_data[np.logical_and(gt_data >= 240, gt_data <= 255)] = 4
    sitk_image = sitk.GetImageFromArray(gt_data)
    sitk.WriteImage(sitk_image, os.path.join(target_path, 'labels', f'{subject_name}_T1DUAL_OutPhase.nii.gz'))
    dcm2nii(os.path.join(file_root_path, subject_name, 'T2SPIR', 'DICOM_anon'), os.path.join(target_path, 'images', f'{subject_name}_T2SPIR.nii.gz'))
    png2nii(os.path.join(file_root_path, subject_name, 'T2SPIR', 'Ground'), os.path.join(target_path, 'labels', f'{subject_name}_T2SPIR.nii.gz'))
    gt_sitk = sitk.ReadImage(os.path.join(target_path, 'labels', f'{subject_name}_T2SPIR.nii.gz'))
    gt_data = sitk.GetArrayFromImage(gt_sitk)
    gt_data[np.logical_and(gt_data >= 55, gt_data <= 70)] = 1
    gt_data[np.logical_and(gt_data >= 110, gt_data <= 135)] = 2
    gt_data[np.logical_and(gt_data >= 175, gt_data <= 200)] = 3
    gt_data[np.logical_and(gt_data >= 240, gt_data <= 255)] = 4
    sitk_image = sitk.GetImageFromArray(gt_data)
    sitk.WriteImage(sitk_image, os.path.join(target_path, 'labels', f'{subject_name}_T2SPIR.nii.gz'))
