import io
import os
import json
import shutil
import numpy as np
import sys

import SimpleITK as sitk

import configparser
from utils.Data_utils import parse_file_tree, copy
from utils.format_convert import dcm2nii, png2nii



################################# organize training set
xlsx_path = '/mnt/ssd1/Datasets/MRI/ATLAS/20220425_ATLAS_2.0_MetaData.xlsx'
import pandas as pd

# Read the Excel file
df = pd.read_excel(xlsx_path)

file_root_path = '/mnt/ssd1/Datasets/MRI/ATLAS/Training'
file_tree = parse_file_tree(file_root_path)

target_path = '/mnt/ssd1/Datasets/MRI/ATLAS/organized/Train'
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
        for session_name, session in subject.items():
            if isinstance(session, dict):
                for session_1_name, session_1 in session.items():
                    nii_files = [f for f in os.listdir(os.path.join(file_root_path, subject_name, session_name, session_1_name, 'anat')) if
                                 f.endswith('.nii') or f.endswith('.nii.gz')]
                    for nii_file in nii_files:
                        if 'label' in nii_file:
                            if not '-L_desc-T1lesion_mask' in nii_file:
                                print(nii_file)

                            gt_file_path = os.path.join(file_root_path, subject_name, session_name, session_1_name, 'anat', nii_file)
                            gt_sitk = sitk.ReadImage(gt_file_path)
                            gt_data = sitk.GetArrayFromImage(gt_sitk)
                            if np.any(np.logical_and(np.unique(gt_data) != 0., np.unique(gt_data) != 0.01)):
                                print(f'have unique: {np.unique(gt_data)}')
                            else:
                                gt_data[gt_data == 0.01] = 1
                                gt_data = gt_data.astype(np.int8)
                            target_gt_file_path = os.path.join(target_path, 'labels', nii_file.split('-MNI152NLin2009aSym')[0]+'.nii.gz')
                            sitk_image = sitk.GetImageFromArray(gt_data)
                            sitk.WriteImage(sitk_image, target_gt_file_path)
                        else:
                            if not '_T1w' in nii_file:
                                print(nii_file)
                            image_file_path = os.path.join(file_root_path, subject_name, session_name, session_1_name, 'anat', nii_file)
                            target_image_file_path = os.path.join(target_path, 'images', nii_file.split('-MNI152NLin2009aSym')[0]+'.nii.gz')
                            copy(image_file_path, target_image_file_path)

################################# organize testing set
file_root_path = '/mnt/ssd1/Datasets/MRI/ATLAS/Testing'
file_tree = parse_file_tree(file_root_path)

target_path = '/mnt/ssd1/Datasets/MRI/ATLAS/organized/Test'
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
        for session_name, session in subject.items():
            if isinstance(session, dict):
                for session_1_name, session_1 in session.items():
                    nii_files = [f for f in os.listdir(os.path.join(file_root_path, subject_name, session_name, session_1_name, 'anat')) if
                                 f.endswith('.nii') or f.endswith('.nii.gz')]
                    for nii_file in nii_files:
                        if 'label' in nii_file:
                            if not '-L_desc-T1lesion_mask' in nii_file:
                                print(nii_file)

                            gt_file_path = os.path.join(file_root_path, subject_name, session_name, session_1_name, 'anat', nii_file)
                            gt_sitk = sitk.ReadImage(gt_file_path)
                            gt_data = sitk.GetArrayFromImage(gt_sitk)
                            if np.any(np.logical_and(np.unique(gt_data) != 0., np.unique(gt_data) != 0.01)):
                                print(f'have unique: {np.unique(gt_data)}')
                            else:
                                gt_data[gt_data == 0.01] = 1
                                gt_data = gt_data.astype(np.int8)
                            target_gt_file_path = os.path.join(target_path, 'labels', nii_file.split('-MNI152NLin2009aSym')[0]+'.nii.gz')
                            sitk_image = sitk.GetImageFromArray(gt_data)
                            sitk.WriteImage(sitk_image, target_gt_file_path)
                        else:
                            if not '_T1w' in nii_file:
                                print(nii_file)
                            image_file_path = os.path.join(file_root_path, subject_name, session_name, session_1_name, 'anat', nii_file)
                            target_image_file_path = os.path.join(target_path, 'images', nii_file.split('-MNI152NLin2009aSym')[0]+'.nii.gz')
                            copy(image_file_path, target_image_file_path)

