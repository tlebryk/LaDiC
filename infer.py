#!/usr/bin/env python3
"""
single_infer.py  —  caption one image with LaDiC

Usage
-----
$ python single_infer.py                         # captions the default image
$ python single_infer.py /path/to/other.jpg      # captions a different image
"""

import argparse, sys, torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

from diff_models.diffcap_model import Diffuser_with_LN
from my_utils.blip_util       import load_checkpoint
from coco_eval                import inference
from configs.config           import MAX_LENGTH, IN_CHANNEL

DEFAULT_IMG = "web.jpg"
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_DIR = 'pretrained_ckpt'

# ───────────────────────── argparse ──────────────────────────
def get_args():
    parser = argparse.ArgumentParser(
        description="Generate a caption for a single image with LaDiC"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMG,
        help=f"Path to the image file (default: {DEFAULT_IMG})"
    )
    return parser.parse_args()

# ──────────────────────── model setup ────────────────────────
def build_model():
    model = Diffuser_with_LN(image_size=224)
    model.visual_encoder, _ = load_checkpoint(
        model.visual_encoder,
        f'{PRETRAINED_DIR}/model_base_capfilt_large.pth'
    )
    state = torch.load(
        "pytorch_model.bin",
        map_location=device
    )
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()

prep = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275 , 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ─────────────────────────── main ────────────────────────────
def main():
    args = get_args()
    img = prep(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    sample = {
        "image": img,
        "attention_mask": torch.ones((1, MAX_LENGTH), device=device),
        "img_id": torch.tensor([0], device=device)
    }

    model = build_model()
    caption, _ = inference(sample, tokenizer, model)
    print("\nCaption:", caption[0])

if __name__ == "__main__":
    main()
