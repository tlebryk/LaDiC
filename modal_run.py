import modal
import subprocess

docker_reg = "tlebryk/ladic:v3.1"
# image = modal.Image.from_registry(docker_reg)
# ]

# Point to the folder that contains your Dockerfile
image = modal.Image.from_dockerfile(
    path="./Dockerfile",  # root of your repo
    # dockerfile="Dockerfile",  # explicit for clarity; default is "Dockerfile"
)
# image.run_commands(
#     "accelerate launch main.py --epoch 2", secrets=["wandb-secret"], gpu="T4"
# )
# stub = modal.Stub("diff-cap-train")

app = modal.App("example-a1111-webui", image=image)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],  # <-- injects env var
    gpu="L4",
    timeout=60 * 60 * 6,
)
def train():
    subprocess.run(
        [
            "accelerate",
            "launch",
            "main.py",
            "--epoch",
            "50",
            "--notes",
            "50_w_pretrain",
            # "--bsz",
            # "4",
            # "--seqlen",
            # "150",
            # "--mixed_precision",
            # "fp16",
            "--full_model_path",
            "pytorch_model.bin",
        ],
        check=True,
    )


# Point to the folder that contains your Dockerfile
image2 = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "beautifulsoup4==4.11.1",
        "einops==0.6.0",
        "evaluate==0.4.0",
        "protobuf==3.20.3",
        "safetensors==0.3.1",
        "scikit-learn==1.2.2",
        "scipy==1.10.1",
        "thinc==8.1.5",
        "threadpoolctl==3.1.0",
        "tqdm==4.64.1",
        "wandb==0.13.4",
        "rouge_score",
        "accelerate==0.20.3",
        "fairscale==0.4.12",
        "timm==0.6.12",
        "bert-score==0.3.13",
        "tokenizers==0.13.1",
        "transformers==4.30.2",
        "gdown",
        "pycocotools",
        "pycocoevalcap",
    )
    .add_local_dir("./", remote_path="./")
    # .add_local_dir("configs", remote_path="/root/configs")
    # .add_local_file("train.json", remote_path="/root/train.json")
    # .add_local_file("val.json", remote_path="/root/val.json")
    # .add_local_file("test.json", remote_path="/root/test.json")
    # .add_local_python_source("llada", "llada_train", "eval_helper")
)


@app.function(
    image=image2,
    secrets=[modal.Secret.from_name("wandb-secret")],  # <-- injects env var
    gpu="L4",
    timeout=60 * 60 * 6,
)
def train2():
    subprocess.run(
        [
            "accelerate",
            "launch",
            "main.py",
            "--epoch",
            "50",
            "--notes",
            "50_w_pretrain",
            # "--bsz",
            # "4",
            # "--seqlen",
            # "150",
            # "--mixed_precision",
            # "fp16",
            "--full_model_path",
            "pytorch_model.bin",
        ],
        check=True,
    )
