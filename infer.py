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
import json
import tqdm
from typing import Dict, Any, List, Optional, Tuple
import evaluate
from datetime import datetime
import numpy as np

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


def run_inference_on_dataset(
    model: torch.nn.Module,
    dataloader,
    tokenizer: BertTokenizer,
    compute_metrics_flag: bool = True,
    skip_bertscores: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Run inference on the entire dataset and collect results"""
    results = []
    all_predictions = []
    all_references = []

    # Load metrics if needed
    metrics_dict = load_metrics() if compute_metrics_flag else None

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            # Move batch to device
            batch_on_device = {
                "image": batch["image"].to(device),
                "attention_mask": torch.ones(
                    (batch["image"].size(0), MAX_LENGTH), device=device
                ),
                "img_id": batch["img_id"],
            }

            # Run inference
            captions, _ = inference(batch_on_device, tokenizer, model)

            # Collect results
            for i in range(len(captions)):
                result_entry = {
                    "img_id": batch["img_id"][i].item(),
                    "true_text": batch["text"][i],
                    "predicted_text": captions[i],
                }

                results.append(result_entry)

                # Collect predictions and references for batch metrics computation
                if compute_metrics_flag:
                    all_predictions.append(captions[i])
                    all_references.append(
                        [batch["text"][i]]
                    )  # References must be a list of lists

    # Compute overall metrics
    overall_metrics = {}
    if compute_metrics_flag and metrics_dict:
        print("Computing evaluation metrics...")
        overall_metrics = compute_metrics(
            all_predictions,
            all_references,
            metrics_dict,
            compute_bertscores=not skip_bertscores,
        )
        print("Metrics computed:")
        for metric_name, metric_value in overall_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

    return results, overall_metrics


def save_results(
    results: List[Dict[str, Any]],
    metrics: Dict[str, float],
    output_path: str,
    run_name: Optional[str] = None,
    model_path: Optional[str] = None,
    dataset_split: Optional[str] = None,
    batch_size: Optional[int] = None,
    notes: Optional[str] = None,
) -> None:
    """Save results and metrics to a JSON file with experiment info"""
    # Create experiment info dictionary
    if run_name is None:
        run_name = os.path.splitext(os.path.basename(output_path))[0]

    experiment_info = {
        "run_name": run_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Add optional metadata if provided
    if model_path:
        experiment_info["model_path"] = model_path
    if dataset_split:
        experiment_info["dataset_split"] = dataset_split
    if batch_size:
        experiment_info["batch_size"] = batch_size
    if notes:
        experiment_info["notes"] = notes

    # Organize output data
    output_data = {
        "experiment_info": experiment_info,
        "metrics": metrics,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_path}")


def load_metrics() -> Dict[str, Any]:
    """Load evaluation metrics from the evaluate package"""
    metrics = {
        "bleu": evaluate.load("bleu"),
        "rouge": evaluate.load("rouge"),
    }

    # BERTScore is more resource-intensive, so we load it conditionally
    try:
        metrics["bertscore"] = evaluate.load("bertscore")
    except Exception as e:
        print(f"Warning: BERTScore could not be loaded: {e}")
        metrics["bertscore"] = None

    return metrics


def compute_metrics(
    predictions: List[str],
    references: List[List[str]],
    metrics: Dict[str, Any],
    compute_bertscores: bool = True,
) -> Dict[str, float]:
    """Compute evaluation metrics for a batch of predictions and references"""
    results = {}

    # BLEU score
    bleu_results = metrics["bleu"].compute(
        predictions=predictions, references=references
    )
    results["bleu"] = bleu_results["bleu"]

    # ROUGE scores
    rouge_results = metrics["rouge"].compute(
        predictions=predictions, references=references
    )
    results["rouge1"] = rouge_results["rouge1"]
    results["rouge2"] = rouge_results["rouge2"]
    results["rougeL"] = rouge_results["rougeL"]

    # BERTScore (optional as it can be slow)
    if compute_bertscores and metrics["bertscore"] is not None:
        try:
            bertscore_results = metrics["bertscore"].compute(
                predictions=predictions, references=references, lang="en"
            )
            results["bertscore_precision"] = np.mean(bertscore_results["precision"])
            results["bertscore_recall"] = np.mean(bertscore_results["recall"])
            results["bertscore_f1"] = np.mean(bertscore_results["f1"])
        except Exception as e:
            print(f"Warning: BERTScore computation failed: {e}")

    return results


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
                "datasets/web/all_data/0B660875-60B4-4E65-9793-3C7EB6C8AFD0.png"
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

    def tmp():
        skip_bertscores = True
        metrics_dict = load_metrics()
        print("Computing evaluation metrics...")
        all_predictions, all_references = ["I am a cat", "That is funny."], [
            "I am a cat though",
            "That is not funny.",
        ]
        overall_metrics = compute_metrics(
            all_predictions,
            all_references,
            metrics_dict,
            compute_bertscores=not skip_bertscores,
        )
        print("Metrics computed:")
        for metric_name, metric_value in overall_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        save_results()
