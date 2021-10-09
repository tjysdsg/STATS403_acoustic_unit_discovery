#!/usr/bin/env bash

set -e
set -u
set -o pipefail

stage=0
zerospeech_path=zerospeech2020_datasets
ckpt=exp/zerospeech_vae/model.pt

. utils/parse_options.sh
. path.sh
. cmd.sh

if [ ${stage} -le 1 ]; then
  python ZeroSpeech/preprocess.py in_dir=${zerospeech_path} dataset=2019/english || exit 1
  python ZeroSpeech/preprocess.py in_dir=${zerospeech_path} dataset=2019/surprise || exit 1
fi

if [ ${stage} -le 2 ]; then
  python ZeroSpeech/encode.py dataset=2019/english out_dir=zerospeech2020_datasets/encode/2019/english/test || exit 1
  python ZeroSpeech/encode.py dataset=2019/surprise out_dir=zerospeech2020_datasets/encode/2019/surprise/test || exit 1
fi
