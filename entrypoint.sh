#!/bin/bash

set -euo pipefail

git clone https://github.com/tlebryk/LaDiC.git
cd LaDiC

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
rm ~/miniconda.sh
conda init
source ~/.bashrc
export PATH=$HOME/miniconda/bin:$PATH

apt update
apt install rustc cargo build-essential

conda env create -f ladic2.yaml
conda activate ladic
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# pip install accelerate
# pip install huggingface_hub
# pip install timm
# pip install einops
# pip install trasfromers
pip install git+https://github.com/openai/CLIP.git

pip install gdown
gdown --id 1oJptCY4oGkMP-LSTIgfx0mFJfMDpH0uG --output pytorch_model.bin
# mv LADic.bin pytorch_model.bin
wget -O cat.jpg https://cataas.com/cat
mkdir -p pretrained_ckpt
wget -O pretrained_ckpt/model_base_capfilt_large.pth https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth

# python infer.py

git pull
mkdir result
wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.zip

# Or if you prefer curl:
# curl -LO https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.zip

# Extract it to the destination
unzip pix2code_datasets.zip -d /app/datasets/COCO
rm pix2code_datasets.zip
python create_ds.py

accelerate launch main.py --epoch 2
