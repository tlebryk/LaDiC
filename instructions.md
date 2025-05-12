## Setting up shell in modal:

```
 pip install modal
  modal setup
```

Run from root
## Download dataset: 
```bash 
pip install gdown
# get the pretrained weights
gdown --id 1oJptCY4oGkMP-LSTIgfx0mFJfMDpH0uG --output pytorch_model.bin 
# get the vision weights
mkdir -p pretrained_ckpt && \
wget -O pretrained_ckpt/model_base_capfilt_large.pth https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth
mkdir -p result 
# get pixel datasets
wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.zip
for i in $(seq -f "%02g" 1 9); do
    wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.z$i
done
zip -F pix2code_datasets.zip --out datasets.zip
unzip datasets.zip -d ./datasets/
# clean up
rm pix2code_datasets.zip
rm pix2code_datasets.z*

```



## Modal set up:
Make sure you have a secret wandb-secret with your wandb key. 


Run training (see readme and configs.py for hyperparameters).
```bash
modal run modal_run.py::train --le
```

Open a shell
```bash
modal shell modal_run.py::train
```

Run training locally: (see readme)
```bash
accelerate launch main.py --epoch 2
```

NOTE: our default max sequence length is 64; the html code is much longer than this... we can run with64 to get a quick baseline and try longer or shorter lengths later. 



