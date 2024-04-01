import io
import os
import json
import shutil
import sys

import SimpleITK as sitk

import configparser
from utils.Data_utils import parse_file_tree, copy
from utils.format_convert import dcm2nii, png2nii

file_root_path = '/mnt/ssd1/Datasets/MRI/picai/Train/labels'
file_tree = parse_file_tree(file_root_path)

target_path = '/mnt/ssd1/Datasets/MRI/picai/Train/labels'

for file_name, file in file_tree.items():
    copy(os.path.join(file_root_path, file_name), os.path.join(target_path, file_name.split('.nii.gz')[0] + '_0000' + '.nii.gz'))
    copy(os.path.join(file_root_path, file_name),
         os.path.join(target_path, file_name.split('.nii.gz')[0] + '_0001' + '.nii.gz'))
    copy(os.path.join(file_root_path, file_name),
         os.path.join(target_path, file_name.split('.nii.gz')[0] + '_0002' + '.nii.gz'))
    os.remove(os.path.join(file_root_path, file_name))