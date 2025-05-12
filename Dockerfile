# =======================
# Base stage (common setup)
# =======================
FROM python:3.10-slim 

# Set working directory
RUN mkdir /LaDiC
WORKDIR /app/LaDiC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    rustc \
    cargo \
    build-essential \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /app/LaDiC
# Add conda to path
# ENV PATH /opt/conda/bin:$PATH

# Create conda environment and install packages
RUN pip install --no-cache-dir -r requirements.txt && \
pip install --no-cache-dir accelerate==0.20.3 fairscale==0.4.12 timm==0.6.12 bert-score==0.3.13 tokenizers==0.13.1 transformers==4.30.2 gdown pycocotools pycocoevalcap
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
# pip install --no-cache-dir torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113 && \

# # # Install additional packages in the conda environment
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git  

# Clone the repository

# # # Download model files
RUN gdown --id 1oJptCY4oGkMP-LSTIgfx0mFJfMDpH0uG --output pytorch_model.bin 
# && \
# wget -O cat.jpg https://cataas.com/cat && \
# mkdir -p pretrained_ckpt && \
# wget -O pretrained_ckpt/model_base_capfilt_large.pth https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth
COPY . /app/LaDiC/

RUN mkdir result
####################################################
# 2) INFERENCE: slim runtime
####################################################
# FROM base AS inference
# # (nothing extra to build — reuses everything from base)
# # default command for inference
# CMD ["python", "infer.py"]

# ####################################################
# # 3) TRAINING (DEFAULT TARGET): add dataset & preprocess
# ####################################################
# FROM base AS training

# fetch & stitch pix2code dataset, then preprocess
# RUN mkdir -p result \
# && wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.zip \
# && for i in $(seq -f "%02g" 1 9); do \
# wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.z$i; \
# done \
# && zip -F pix2code_datasets.zip --out datasets.zip \
# && unzip datasets.zip -d ./datasets/COCO/ \
# && python create_ds.py


CMD ["accelerate", "launch", "main.py"]