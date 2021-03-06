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

## ABX testing

0. Prepare python environment as described [here](https://github.com/tjysdsg/zerospeech2020)
1. Run `preprocess_and_encode_zerospeech.sh`, results are saved in `zerospeech_data`
2. Run

```
ZEROSPEECH2020_DATASET=/mnt/e/datasets/zerospeech2020/2020 \
  zerospeech2020-evaluate 2019 -j10 zerospeech2020_datasets/submission
```

### Results

#### L1 only

```json5
{
  "2019": {
    "english": {
      "scores": {
        "abx": 39.86074224642665,
        "bitrate": 387.90274352928196
      },
      "details_bitrate": {
        "test": 387.90274352928196
      },
      "details_abx": {
        "test": {
          "cosine": 39.86074224642665,
          "KL": 50.0,
          "levenshtein": 42.31439953566402
        }
      }
    }
  }
}
```

#### L1+L2

```json5
{
  "2019": {
    "english": {
      "scores": {
        "abx": 40.45447005042622,
        "bitrate": 404.3762622526002
      },
      "details_bitrate": {
        "test": 404.3762622526002
      },
      "details_abx": {
        "test": {
          "cosine": 40.45447005042622,
          "KL": 50.0,
          "levenshtein": 42.71491717354263
        }
      }
    }
  }
}
```