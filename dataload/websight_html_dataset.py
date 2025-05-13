import os
import json

from torch.utils.data import Dataset
from PIL import Image

from dataload.utils import pre_caption # Assuming pre_caption can handle HTML strings or will be a no-op
from configs.config import *


class websight_html_train(Dataset):
    def __init__(
        self, transform, tokenizer, image_root, ann_root, max_words=MAX_LENGTH, prompt="" # max_words usually from config
    ):
        """
        image_root (string): Root directory of images (e.g. datasets/websight/all_data)
        ann_root (string): directory to store the annotation file (e.g. datasets/websight/)
        """
        filename = "train.json"
        self.annotation = json.load(open(os.path.join(ann_root, filename), "r", encoding='utf-8'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words # For HTML, this might be less relevant if pre_caption is a no-op
        self.prompt = prompt
        self.tokenizer = tokenizer

        # Create a simple mapping if needed, or remove if img_ids are not strictly used downstream
        # For now, keeping it similar to COCO, but it might not be essential
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"] # Assumes 'image_id' exists in your JSON
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

        # For HTML, the 'caption' is the HTML string.
        # pre_caption might need to be adjusted or be a no-op.
        # If pre_caption expects a list of strings (like COCO), and ann["caption"] is a single HTML string,
        # it might work if pre_caption handles single strings, or it might need ann["caption"] = [ann["caption"]]
        # For now, let's assume pre_caption can handle a single string.
        html_content = ann["caption"] # This is the HTML string

        # The pre_caption function might truncate. For HTML, we usually want the whole thing for the tokenizer.
        # Consider if pre_caption is necessary or if direct tokenization of html_content is better.
        # For now, passing it through pre_caption as in the original.
        processed_html = pre_caption(html_content, self.max_words, is_html=True)

        tokens = self.tokenizer(
            text=processed_html, # Tokenize the (potentially pre-processed) HTML
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH, # Ensure this is the HTML tokenizer's max length
        )
        # print("tokens", tokens) # Remove or comment out this debug line
        return {
            "image": image,
            "text": processed_html, # Store the (potentially pre-processed) HTML text
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
        }


class websight_html_eval(Dataset): # Simplified inheritance, __init__ is mostly distinct
    def __init__(
        self,
        transform,
        tokenizer,
        image_root,
        ann_root,
        max_words=MAX_LENGTH, # max_words usually from config
        prompt="",
        split="val",
    ):
        """
        image_root (string): Root directory of images (e.g. datasets/websight/all_data)
        ann_root (string): directory to store the annotation file (e.g. datasets/websight/)
        split (string): 'val' or 'test'
        """
        filename = f"{split}.json"
        self.annotation = json.load(open(os.path.join(ann_root, filename), "r", encoding='utf-8'))
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
        
        # Ensure all annotations have an image_id for the eval case
        # This was missing in the original coco_karpathy_caption_eval's __init__ but present in its __getitem__
        # For consistency and if used by evaluation metrics:
        self.test_annotation = []
        for ann in self.annotation:
            self.test_annotation.append(ann)


    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        html_content = ann["caption"] # This is the HTML string
        processed_html = pre_caption(html_content, self.max_words, is_html=True)

        tokens = self.tokenizer(
            text=processed_html,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH, # Ensure this is the HTML tokenizer's max length
        )
        
        return {
            "image": image,
            "text": processed_html, # Store the (potentially pre-processed) HTML text
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "img_id": ann["image_id"], # Make sure 'image_id' is in your JSONs
        } 