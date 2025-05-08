#!/bin/bash

set -euo pipefail

git clone https://github.com/tlebryk/LaDiC.git
cd LaDiC

# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
# bash ~/miniconda.sh -b -p $HOME/miniconda
# rm ~/miniconda.sh
# conda init
# source ~/.bashrc
# export PATH=$HOME/miniconda/bin:$PATH

apt update
apt install -y rustc cargo build-essential unzip zip

# conda env create -f ladic2.yaml
# conda activate ladic
# conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# pip install accelerate
# pip install huggingface_hub
# pip install timm
# pip install einops
# pip install trasfromers
cd /app
pip install -r requirements.txt
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
wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.z01
wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.z02
wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.z03
wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.z04
wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.z05
wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.z06
wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.z07
wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.z08
wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.z09
zip -F pix2code_datasets.zip --out datasets.zip
unzip datasets.zip -d /app/datasets/COCO/
python create_ds.py

# rm pix2code_datasets.zip
accelerate launch main.py --epoch 2
