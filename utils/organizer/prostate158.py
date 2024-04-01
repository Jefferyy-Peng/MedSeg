import os
import shutil
import numpy as np
import pandas as pd
import yaml
import SimpleITK as sitk
import munch
from utils.Data_utils import parse_file_tree, copy

# from typing import Union
# from monai.utils import first
# from functools import partial
# from collections import namedtuple
# from monai.data import DataLoader as MonaiDataLoader

train = True
valid = True
test = True

def load_config(fn: str = 'config.yaml'):
    "Load config from YAML and return a serialized dictionary object"
    with open(fn, 'r') as stream:
        config = yaml.safe_load(stream)
    config = munch.munchify(config)

    if not config.overwrite:
        i = 1
        while os.path.exists(config.run_id + f'_{i}'):
            i += 1
        config.run_id += f'_{i}'

    config.out_dir = os.path.join(config.run_id, config.out_dir)
    config.log_dir = os.path.join(config.run_id, config.log_dir)

    if not isinstance(config.data.image_cols, (tuple, list)):
        config.data.image_cols = [config.data.image_cols]
    if not isinstance(config.data.label_cols, (tuple, list)):
        config.data.label_cols = [config.data.label_cols]

    config.transforms.mode = ('bilinear',) * len(config.data.image_cols) + \
                             ('nearest',) * len(config.data.label_cols)
    return config

config_fn = '../anatomy.yaml'
config = load_config(config_fn)

## parse needed rguments from config
if train is None: train = config.data.train
if valid is None: valid = config.data.valid
if test is None: test = config.data.test

data_dir = config.data.data_dir
train_csv = config.data.train_csv
valid_csv = config.data.valid_csv
test_csv = config.data.test_csv
image_cols = config.data.image_cols
label_cols = config.data.label_cols
dataset_type = config.data.dataset_type
cache_dir = config.data.cache_dir
batch_size = config.data.batch_size
debug = config.debug

## ---------- data dicts ----------

# first a global data dict, containing only the filepath from image_cols and label_cols is created. For this,
# the dataframe is reduced to only the relevant columns. Then the rows are iterated, converting each row into an
# individual dict, as expected by monai

if not isinstance(image_cols, (tuple, list)): image_cols = [image_cols]
if not isinstance(label_cols, (tuple, list)): label_cols = [label_cols]

train_df = pd.read_csv(train_csv)
valid_df = pd.read_csv(valid_csv)
test_df = pd.read_csv(test_csv)
if debug:
    train_df = train_df.sample(25)
    valid_df = valid_df.sample(5)

train_df['split'] = 'train'
valid_df['split'] = 'valid'
test_df['split'] = 'test'
whole_df = []
if train: whole_df += [train_df]
if valid: whole_df += [valid_df]
if test: whole_df += [test_df]
df = pd.concat(whole_df)
cols = image_cols + label_cols
for col in cols:
    # create absolute file name from relative fn in df and data_dir
    df[col] = [os.path.join(data_dir, fn) for fn in df[col]]
    if not os.path.exists(list(df[col])[0]):
        raise FileNotFoundError(list(df[col])[0])

data_dict = [dict(row[1]) for row in df[cols].iterrows()]
# data_dict is not the correct name, list_of_data_dicts would be more accurate, but also longer.
# The data_dict looks like this:
# [
#  {'image_col_1': 'data_dir/path/to/image1',
#   'image_col_2': 'data_dir/path/to/image2'
#   'label_col_1': 'data_dir/path/to/label1},
#  {'image_col_1': 'data_dir/path/to/image1',
#   'image_col_2': 'data_dir/path/to/image2'
#   'label_col_1': 'data_dir/path/to/label1},
#    ...]
# Filename should now be absolute or relative to working directory

# now we create separate data dicts for train, valid and test data respectively
assert train or test or valid, 'No dataset type is specified (train/valid or test)'

if test:
    test_files = list(map(data_dict.__getitem__, *np.where(df.split == 'test')))

if valid:
    val_files = list(map(data_dict.__getitem__, *np.where(df.split == 'valid')))

if train:
    train_files = list(map(data_dict.__getitem__, *np.where(df.split == 'train')))

source_path = '/mnt/ssd1/Datasets/MRI/prostate158'
target_path = '/mnt/ssd1/Datasets/MRI/prostate158/organized'

os.makedirs(os.path.join(target_path, 'Train', 'images'), exist_ok=True)
os.makedirs(os.path.join(target_path, 'Train', 'labels'), exist_ok=True)
os.makedirs(os.path.join(target_path, 'Test', 'images'), exist_ok=True)
os.makedirs(os.path.join(target_path, 'Test', 'labels'), exist_ok=True)

for subject in train_files:
    for modal_name, modal_path in subject.items():
        if 'adc_tumor_reader' in modal_name:
            adc_gt_sitk = sitk.ReadImage(modal_path)
            adc_data = sitk.GetArrayFromImage(adc_gt_sitk)
            adc_data[adc_data == 1] = 3
            sitk_image = sitk.GetImageFromArray(adc_data)
            sitk.WriteImage(sitk_image, os.path.join(target_path, 'Train', 'labels', modal_path.split('/')[7] + '_' + modal_name.split('_')[0] + '.nii.gz'))
        elif 't2_anatomy_reader' in modal_name:
            copy(modal_path, os.path.join(target_path, 'Train', 'labels', modal_path.split('/')[7] + '_' + modal_name.split('_')[0] + '.nii.gz'))
        else:
            copy(modal_path, os.path.join(target_path, 'Train', 'images', modal_path.split('/')[7] + '_' + modal_path.split('/')[8]))

for subject in val_files:
    for modal_name, modal_path in subject.items():
        if 'adc_tumor_reader' in modal_name:
            adc_gt_sitk = sitk.ReadImage(modal_path)
            adc_data = sitk.GetArrayFromImage(adc_gt_sitk)
            adc_data[adc_data == 1] = 3
            sitk_image = sitk.GetImageFromArray(adc_data)
            sitk.WriteImage(sitk_image, os.path.join(target_path, 'Train', 'labels',
                                                     modal_path.split('/')[7] + '_' + modal_name.split('_')[
                                                         0] + '.nii.gz'))
        elif 't2_anatomy_reader' in modal_name:
            copy(modal_path, os.path.join(target_path, 'Train', 'labels',
                                          modal_path.split('/')[7] + '_' + modal_name.split('_')[0] + '.nii.gz'))
        else:
            copy(modal_path, os.path.join(target_path, 'Train', 'images',
                                          modal_path.split('/')[7] + '_' + modal_path.split('/')[8]))

for subject in test_files:
    for modal_name, modal_path in subject.items():
        if 'adc_tumor_reader' in modal_name:
            adc_gt_sitk = sitk.ReadImage(modal_path)
            adc_data = sitk.GetArrayFromImage(adc_gt_sitk)
            adc_data[adc_data == 1] = 3
            sitk_image = sitk.GetImageFromArray(adc_data)
            sitk.WriteImage(sitk_image, os.path.join(target_path, 'Test', 'labels',
                                                     modal_path.split('/')[7] + '_' + modal_name.split('_')[
                                                         0] + '.nii.gz'))
        elif 't2_anatomy_reader' in modal_name:
            copy(modal_path, os.path.join(target_path, 'Test', 'labels',
                                          modal_path.split('/')[7] + '_' + modal_name.split('_')[0] + '.nii.gz'))
        else:
            copy(modal_path, os.path.join(target_path, 'Test', 'images',
                                          modal_path.split('/')[7] + '_' + modal_path.split('/')[8]))
