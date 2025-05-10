# infer.py
"""
single_infer.py  —  caption one image with LaDiC

Usage
-----
$ python single_infer.py                         # captions the default image
$ python single_infer.py /path/to/other.jpg      # captions a different image
"""
import os
import argparse, sys, torch

from transformers import BertTokenizer

from diff_models.diffcap_model import Diffuser_with_LN
from diff_models.diffusion import *

from my_utils.blip_util import load_checkpoint
from coco_eval import inference

# from configs.config           import MAX_LENGTH, IN_CHANNEL

DEFAULT_IMG = "web.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_DIR = "pretrained_ckpt"
MAX_LENGTH = 24
IN_CHANNEL = 768


# ──────────────────────── model setup ────────────────────────
def build_model():
    model = Diffuser_with_LN(image_size=224)
    model.visual_encoder, _ = load_checkpoint(
        model.visual_encoder, f"{PRETRAINED_DIR}/model_base_capfilt_large.pth"
    )
    if FULL_MODEL_PATH:
        state = torch.load(FULL_MODEL_PATH, map_location=device)
    else:
        # final_model_dir = f"{LOG_DIR}/{MODEL_NAME}"
        # state = torch.load(
        #     os.path.join(final_model_dir, "pytorch_model.bin"), map_location=device
        # )
        state = torch.load("pytorch_model.bin", map_location=device)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


# ─────────────────────────── main ────────────────────────────
def main():
    # args = get_args()
    model = build_model()
    model.eval().to(device)
    # %%
    from PIL import Image
    from torchvision import transforms

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    prep = transforms.Compose(
        [
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    img = (
        prep(
            Image.open(
                "datasets/COCO/web/all_data/024D233C-5B4A-4096-9145-94C2CCDEF0CD.png"
            ).convert("RGB")
        )
        .unsqueeze(0)
        .to(device)
    )

    img = prep(Image.open(IMAGE_PATH).convert("RGB")).unsqueeze(0).to(device)

    sample = {
        "image": img,
        "attention_mask": torch.ones((1, MAX_LENGTH), device=device),
        "img_id": torch.tensor([0], device=device),
    }

    caption, _ = inference(sample, tokenizer, model)
    print("\nCaption:", caption[0])
    # %%


if __name__ == "__main__":
    main()
