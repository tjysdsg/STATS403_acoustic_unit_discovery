# Quick start

## Dependencies

0. Install pytorch https://pytorch.org/get-started
1. `pip install -r requirements.txt`
2. Follow the instruction [here](https://github.com/NVIDIA/apex#linux) to install nvidia apex, note that only the
   python-build is required i.e. `pip install -v --disable-pip-version-check --no-cache-dir ./`

## Scripts

1. Run `preprocess.sh` to extract spectrogram features, saved in `datasets/feats`
2. Run `python vq_vae/train.py`, models are saved in `exp/zerospeech_vae`
3. Run `python vq_vae/encode.py ++checkpoint=<checkpoint_path>`, outputs are saved in `exp/zerospeech_vae`
    - (Optional) Run `python vq_vae/visualize_encodings.py` to plot VQ-VAE encoding outputs, saved
      in `exp/zerospeech_vae/encode`
