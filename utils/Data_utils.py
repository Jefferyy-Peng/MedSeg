import pickle

from skimage.measure import find_contours
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform
import torch
from matplotlib.animation import FuncAnimation, FFMpegWriter
import multiprocessing
import random
import shutil

from tqdm import tqdm

from utils.sem_seg_dataset import SemSegDataset


def copy(source_file_path, destination_path):
    try:
        shutil.copy(source_file_path, destination_path)
        print(f"File copied successfully to {destination_path}")
    except IOError as e:
        print(f"Unable to copy file. {e}")
    except:
        print(f"Unexpected error:", sys.exc_info())

def parse_file_tree(directory):
    result = {}
    for entry in os.scandir(directory):
        if entry.is_file():
            result[entry.name] = None
        elif entry.is_dir():
            result[entry.name] = parse_file_tree(entry.path)
    return result

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }

class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ACDC||AMOS||ATLAS||CHAO||prostate158||picai",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        # tokenizer,
                        # vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                    )
                )
            # elif dataset == "refer_seg":
            #     self.all_datasets.append(
            #         ReferSegDataset(
            #             base_image_dir,
            #             tokenizer,
            #             vision_tower,
            #             samples_per_epoch,
            #             precision,
            #             image_size,
            #             num_classes_per_sample,
            #             exclude_val,
            #             refer_seg_data,
            #         )
            #     )
            # elif dataset == "vqa":
            #     self.all_datasets.append(
            #         VQADataset(
            #             base_image_dir,
            #             tokenizer,
            #             vision_tower,
            #             samples_per_epoch,
            #             precision,
            #             image_size,
            #             num_classes_per_sample,
            #             exclude_val,
            #             vqa_data,
            #         )
            #     )
            # elif dataset == "reason_seg":
            #     self.all_datasets.append(
            #         ReasonSegDataset(
            #             base_image_dir,
            #             tokenizer,
            #             vision_tower,
            #             samples_per_epoch,
            #             precision,
            #             image_size,
            #             num_classes_per_sample,
            #             exclude_val,
            #             reason_seg_data,
            #             explanatory,
            #         )
            #     )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference

class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(self.base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )


class PICAI_DATASET:
    def __init__(self, data_root, batch_size, export_path, dataset_percentage=1, is_regenerate_batches=True, is_write_seperately=False):
        self.Data_root = data_root
        self.batch_size = batch_size
        if not os.path.exists(export_path):
            os.mkdir(export_path)
        self.dataset_percentage = dataset_percentage
        self.export_path = export_path
        self.index = 0
        self._t2w_mri = None
        self._adc_mri = None
        self._dwi_mri = None
        self._gt_label = None
        self.experiments_with_annotation = None
        self.initialize_picai_info()
        self.is_write_seperately = is_write_seperately
        if is_regenerate_batches:
            if is_write_seperately:
                self._preprocess_data_batch_and_write_seperately()
            else:
                self.data = self._preprocess_data_batch_and_write()
        else:
            if is_write_seperately:
                pass
            else:
                self.data = pickle.load(open(os.path.join(self.export_path, f'data_sorted_size.p'), 'rb'))

    def _read_nii_image(self, file_path, dtype=np.uint16):
        return nib.load(file_path).get_fdata().astype(dtype)

    def _read_nii_image_batch(self, file_paths, dtype=np.uint16):
        data_list = []
        for sample in file_paths:
            modal_list = []
            for channle in sample:
                this_data = self._read_nii_image(channle, dtype=dtype)
                if this_data.shape[:2] != (1024, 1024):
                    this_data = transform.resize(
                        this_data, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
                    ).astype(np.uint16)
                modal_list.append(this_data)
            modals = np.stack(modal_list)
            data_list.append(modals.swapaxes(3,0))
        data = np.concatenate(data_list)
        return data_list


    def _normalize_image(self, image):
        return (image - np.min(image)) / (np.max(image) - np.min(image))

    def initialize_picai_info(self):
        '''
        This function finds all experiments with annotation and store the path of those files to a list
        '''
        image_dir = os.path.join(self.Data_root, 'imagesTr')
        anno_dir = os.path.join(self.Data_root, 'labelsTr')
        data_files = os.listdir(image_dir)
        subject_run_dict = {}
        overall_count = 0
        count = 0
        self.experiments_with_annotation = []
        for data_file in data_files:
            overall_count += 1
            subject_id = data_file.split('_')[0]
            run_id = data_file.split('_')[1]
            if subject_id not in subject_run_dict.keys():
                subject_run_dict[subject_id] = []
                subject_run_dict[subject_id].append(run_id)
            if run_id not in subject_run_dict[subject_id]:
                subject_run_dict[subject_id].append(run_id)
            anno_path = os.path.join(anno_dir, subject_id + '_' + run_id + '.nii.gz')
            t2w_path = os.path.join(image_dir, subject_id + '_' + run_id + '_0000.nii.gz')
            adc_path = os.path.join(image_dir, subject_id + '_' + run_id + '_0001.nii.gz')
            dwi_path = os.path.join(image_dir, subject_id + '_' + run_id + '_0002.nii.gz')
            if os.path.exists(anno_path):
                self.experiments_with_annotation.append([t2w_path, adc_path, dwi_path, anno_path])
            else:
                count += 1
        self.experiments_with_annotation = np.array(self.experiments_with_annotation)
        print(f'There are {count} out of {overall_count} ({count/overall_count*100}%) experiments lack annotation')

    def _preprocess_data_batch_and_write_seperately(self, dtype=np.uint16):
        iter_num = int(len(self.experiments_with_annotation) * self.dataset_percentage / self.batch_size)
        actual_idx = 0
        print('processing data into batches')
        pbar = tqdm(total=len(self.experiments_with_annotation), desc=f'Preprocessing dataset', unit='batch')
        pbar.update(mini_batch_i := 0)
        for i in range(iter_num):
            file_path = os.path.join(self.export_path, f'batch_{i}.p')
            if not os.path.exists(file_path):
                data_dict = {}
                for sample in self.experiments_with_annotation[i * self.batch_size:(i+1) * self.batch_size]:
                    modal_list = []
                    label = sample[-1]
                    this_label = self._read_nii_image(label, dtype=dtype).swapaxes(2, 0)
                    if this_label.max() == 0:
                        continue
                    for channle in sample[:3]:
                        this_data = self._read_nii_image(channle, dtype=dtype).swapaxes(2,0)
                        modal_list.append(this_data)
                    modal_list.append(this_label)
                    modals = np.stack(modal_list).transpose(1, 0, 2, 3)
                    if modals.shape[2] not in data_dict.keys():
                        data_dict[modals.shape[2]] = []
                        data_dict[modals.shape[2]].append(modals)
                    else:
                        data_dict[modals.shape[2]].append(modals)
                for shape, data_arrays in data_dict.items():
                    data = np.concatenate(data_arrays)
                    data_dict[shape] = data
                if data_dict:
                    pickle.dump(data_dict, open(file_path, 'wb'))
                actual_idx += 1
                pbar.update(1)

    def _preprocess_data_batch_and_write(self, dtype=np.uint16):
        iter_num = int(len(self.experiments_with_annotation) * self.dataset_percentage / self.batch_size)
        data_dict = {}
        count = 0
        file_path = os.path.join(self.export_path, f'data_sorted_size.p')
        print('processing data into batches')
        pbar = tqdm(total=len(self.experiments_with_annotation), desc=f'Preprocessing dataset', unit='batch')
        pbar.update(mini_batch_i := 0)
        for sample in self.experiments_with_annotation:
            modal_list = []
            label = sample[-1]
            this_label = self._read_nii_image(label, dtype=dtype).swapaxes(2, 0)
            if this_label.max() == 0:
                count += 1
                pbar.update(1)
                continue
            for channle in sample[:3]:
                this_data = self._read_nii_image(channle, dtype=dtype).swapaxes(2, 0)
                modal_list.append(this_data)
            modal_list.append(this_label)
            modals = np.stack(modal_list).transpose(1, 0, 2, 3)
            if modals.shape[2] not in data_dict.keys():
                data_dict[modals.shape[2]] = []
                data_dict[modals.shape[2]].append(modals)
            else:
                data_dict[modals.shape[2]].append(modals)
            pbar.update(1)
        idx = []
        for shape, data_arrays in data_dict.items():
            data = np.concatenate(data_arrays)
            data_dict[shape] = data.transpose(0,2,3,1)
            for value_idx, _ in enumerate(data):
                idx.append((shape, value_idx))
        random.shuffle(idx)
        data_dict['idx'] = idx
        pickle.dump(data_dict, open(file_path, 'wb'))
        return data_dict


        # t2w_mri_list = []
        # adc_mri_list = []
        # dwi_mri_list = []
        # gt_label_list = []
        # overall_count = 0
        # count = 0
        # for subject_id in subject_run_dict.keys():
        #     for run_id in subject_run_dict[subject_id]:
        #         overall_count += 1
        #         anno_path = os.path.join(anno_dir, subject_id + '_' + run_id + '.nii.gz')
        #         if os.path.exists(anno_path):
        #             t2w_mri = self._read_nii_image(os.path.join(image_dir, subject_id + '_' + run_id + '_0000.nii.gz'))
        #             adc_mri = self._read_nii_image(os.path.join(image_dir, subject_id + '_' + run_id + '_0001.nii.gz'))
        #             dwi_mri = self._read_nii_image(os.path.join(image_dir, subject_id + '_' + run_id + '_0002.nii.gz'))
        #             mri_comb = np.stack([t2w_mri, adc_mri, dwi_mri])
        #             t2w_mri_list.append(t2w_mri)
        #             adc_mri_list.append(adc_mri)
        #             dwi_mri_list.append(dwi_mri)
        #             gt_label_list.append(self._read_nii_image(anno_path))
        #         else:
        #             count += 1
        #             print(f'Experiment of subject {subject_id}, run {run_id} does not have annotation, ignore this run')
        # print(f'There are {count} out of {overall_count} ({count/overall_count*100}%) experiments lack annotation')
        # self._t2w_mri = np.stack(t2w_mri_list)
        # self._store_packaged_data(self._t2w_mri, 't2w.p')
        # self._adc_mri = np.stack(adc_mri_list)
        # self._dwi_mri = np.stack(dwi_mri_list)
        # self._gt_label = np.stack(gt_label_list)

    def __len__(self):
        return int(len(self.experiments_with_annotation) * self.dataset_percentage / self.batch_size)

    def __iter__(self):
        return self
    def __next__(self):
        # if self.index >= int(len(self.experiments_with_annotation) * self.dataset_percentage / self.batch_size):
        if self.is_write_seperately:
            if self.index >= len(os.listdir('/data/leo/drive1/Datasets/picai/packaged_data')):
                self.index = 0
                raise StopIteration

            data = pickle.load(open(os.path.join(self.export_path, f'batch_{self.index}.p'), 'rb'))
            self.index += 1
            return self.index, data
        else:
            if self.index >= int(len(self.data['idx']) / self.batch_size):
                self.index = 0
                raise StopIteration
            batch = []
            for size, idx in self.data['idx'][self.index * self.batch_size : (self.index + 1) * self.batch_size]:
                img = self.data[size][idx]
                if img.shape[1] != 1024:
                    img = transform.resize(
                    img, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
                ).astype(np.float32)
                batch.append(img)
            self.index += 1
            return np.concatenate(batch)


