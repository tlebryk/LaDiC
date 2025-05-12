import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from dataload.utils import pre_caption
from configs.config import *


class coco_karpathy_train(Dataset):
    def __init__(
        self, transform, tokenizer, image_root, ann_root, max_words=30, prompt=""
    ):
        """
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        url = "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json"
        filename = "train.json"

        download_url(url, ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, filename), "r"))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.tokenizer = tokenizer

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption = pre_caption(ann["caption"], self.max_words)
        tokens = self.tokenizer(
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        # return image, caption, self.img_ids[ann['image_id']]
        return {
            "image": image,
            "text": caption,
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
        }


class coco_karpathy_caption_eval(coco_karpathy_train):
    def __init__(
        self,
        transform,
        tokenizer,
        image_root,
        ann_root,
        max_words=30,
        prompt="",
        split="val",
    ):
        """
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        url = "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json"
        filename = f"{split}.json"
        download_url(url, ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, filename), "r"))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.tokenizer = tokenizer

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption = pre_caption(ann["caption"], self.max_words)
        tokens = self.tokenizer(
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        # return image, caption, self.img_ids[ann['image_id']]
        return {
            "image": image,
            "text": caption,
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "img_id": ann["image_id"],
        }
