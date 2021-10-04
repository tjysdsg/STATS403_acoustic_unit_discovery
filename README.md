# Quick start

1. Run `preprocess.sh` to extract spectrogram features, saved in `datasets/feats`
2. Run `python vq_vae/train.py`, models are saved in `exp/zerospeech_vae`
3. Run `python vq_vae/encode.py ++checkpoint=<checkpoint_path>`, outputs are saved in `exp/zerospeech_vae`
    - (Optional) Run `python vq_vae/visualize_encodings.py` to plot VQ-VAE encoding outputs, saved
      in `exp/zerospeech_vae/encode`
