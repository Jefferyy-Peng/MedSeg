import glob
import json
import os
import random

import numpy as np
import torch

from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import SHORT_QUESTION_LIST, ANSWER_LIST, ACDC_CLASSES, AMOS_CLASSES, picai_CLASSES, ATLAS_CLASSES, \
    CHAO_CLASSES, prostate158_CLASSES


def init_ACDC(base_image_dir):
    ACDC_data_root = os.path.join(base_image_dir, "ACDC")
    ACDC_classes = ACDC_CLASSES
    ACDC_labels = sorted(
        glob.glob(
            os.path.join(ACDC_data_root, "preprocessed", "Train", "labels", "*.npy")
        )
    )
    ACDC_images = [
        x.replace("labels", "images")
        for x in ACDC_labels
    ]
    print("ACDC: ", len(ACDC_images))
    return ACDC_classes, ACDC_images, ACDC_labels

def init_AMOS(base_image_dir):
    AMOS_data_root = os.path.join(base_image_dir, "AMOS")
    AMOS_classes = AMOS_CLASSES
    AMOS_labels = sorted(
        glob.glob(
            os.path.join(AMOS_data_root, "preprocessed", "Train", "labels", "*.npy")
        )
    )
    AMOS_images = [
        x.replace("labels", "images")
        for x in AMOS_labels
    ]
    print("AMOS: ", len(AMOS_images))
    return AMOS_classes, AMOS_images, AMOS_labels

def init_picai(base_image_dir):
    picai_data_root = os.path.join(base_image_dir, "picai")
    picai_classes = picai_CLASSES
    picai_labels = sorted(
        glob.glob(
            os.path.join(picai_data_root, "preprocessed", "Train", "labels", "*.npy")
        )
    )
    picai_images = [
        x.replace("labels", "images")
        for x in picai_labels
    ]
    print("picai: ", len(picai_images))
    return picai_classes, picai_images, picai_labels

def init_prostate158(base_image_dir):
    prostate158_data_root = os.path.join(base_image_dir, "prostate158")
    prostate158_classes = prostate158_CLASSES
    prostate158_labels = sorted(
        glob.glob(
            os.path.join(prostate158_data_root, "preprocessed", "Train", "labels", "*.npy")
        )
    )
    prostate158_images = [
        x.replace("labels", "images")
        for x in prostate158_labels
    ]
    print("prostate158: ", len(prostate158_images))
    return prostate158_classes, prostate158_images, prostate158_labels

def init_CHAO(base_image_dir):
    CHAO_data_root = os.path.join(base_image_dir, "CHAO")
    CHAO_classes = CHAO_CLASSES
    CHAO_labels = sorted(
        glob.glob(
            os.path.join(CHAO_data_root, "preprocessed", "Train", "labels", "*.npy")
        )
    )
    CHAO_images = [
        x.replace("labels", "images")
        for x in CHAO_labels
    ]
    print("CHAO: ", len(CHAO_images))
    return CHAO_classes, CHAO_images, CHAO_labels

def init_ATLAS(base_image_dir):
    ATLAS_data_root = os.path.join(base_image_dir, "ATLAS")
    ATLAS_classes = ATLAS_CLASSES
    ATLAS_labels = sorted(
        glob.glob(
            os.path.join(ATLAS_data_root, "preprocessed", "Train", "labels", "*.npy")
        )
    )
    ATLAS_images = [
        x.replace("labels", "images")
        for x in ATLAS_labels
    ]
    print("ATLAS: ", len(ATLAS_images))
    return ATLAS_classes, ATLAS_images, ATLAS_labels

class SemSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        # tokenizer,
        # vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        sem_seg_data="ATLAS||ACDC||CHAO||prostate158||AMOS||picai",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        # self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        # self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        for ds in self.sem_seg_datas:
            classes, images, labels = eval("init_{}".format(ds))(base_image_dir)
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes

    def __len__(self):
        return self.samples_per_epoch

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
        ds = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds]

        if ds in ["ACDC", "AMOS", "picai", "prostate158", "ATLAS", "CHAO"]:
            class_map = self.data2classes[ds]
            img_ids, lbl = self.data2list[ds]
        elif ds in ["paco_lvis", "pascal_part"]:
            class_map = self.data2classes[ds]
            img_ids, coco_api = self.data2list[ds]
            idx = random.randint(0, len(img_ids) - 1)
            img_id = img_ids[idx]
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            if ds == "pascal_part":
                file_name = os.path.join(
                    "VOCdevkit", "VOC2010", "JPEGImages", file_name
                )
                image_path = os.path.join(self.base_image_dir, "vlpart", ds, file_name)
            elif ds == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "coco", file_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]
            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)
            if len(anns) == 0:
                return self.__getitem__(0)
            if len(anns) >= self.num_classes_per_sample:
                sampled_anns = np.random.choice(
                    anns, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_anns = anns
            sampled_classes = []
            for ann in sampled_anns:
                sampled_cls = class_map[ann["category_id"]]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    if random.random() < 0.5:
                        name = obj + " " + part
                    else:
                        name = "the {} of the {}".format(part, obj)
                else:
                    name = sampled_cls
                sampled_classes.append(name)

        elif ds in ["ade20k", "cocostuff", "mapillary"]:
            image, labels = self.data2list[ds]
            idx = random.randint(0, len(image) - 1)
            image_path = image[idx]
            label_path = labels[idx]
            label = Image.open(label_path)
            label = np.array(label)
            if ds == "ade20k":
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif ds == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255
            img = cv2.imread(image_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
            image = self.transform.apply_image(image)  # preprocess image for sam
            resize = image.shape[:2]
            unique_label = np.unique(label).tolist()
            if 255 in unique_label:
                unique_label.remove(255)
            if len(unique_label) == 0:
                return self.__getitem__(0)

            classes = [self.data2classes[ds][class_id] for class_id in unique_label]
            if len(classes) >= self.num_classes_per_sample:
                sampled_classes = np.random.choice(
                    classes, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_classes = classes

        questions = []
        answers = []
        class_ids = []
        for sampled_cls in sampled_classes:
            text = sampled_cls

            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))

            answers.append(random.choice(self.answer_list))

            if ds in ["paco_lvis", "pascal_part"]:
                continue

            class_id = self.data2classes[ds].tolist().index(sampled_cls)
            class_ids.append(class_id)

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if ds in ["paco_lvis", "pascal_part"]:
            masks = []
            for ann in sampled_anns:
                try:
                    masks.append(coco_api.annToMask(ann))
                except Exception as e:
                    print(e)
                    return self.__getitem__(0)

            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        else:
            label = torch.from_numpy(label).long()
            masks = []
            for class_id in class_ids:
                masks.append(label == class_id)
            masks = torch.stack(masks, dim=0)
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )