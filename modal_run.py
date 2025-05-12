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
