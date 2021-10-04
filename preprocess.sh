#!/usr/bin/env bash

set -e
set -u
set -o pipefail

stage=0

. utils/parse_options.sh
. path.sh
. cmd.sh

if [ ${stage} -le 1 ]; then
  python local/l2_arctic.py --l2-path=/home/storage15/tangjiyang/data/l2arctic --output-dir=datasets || exit 1
fi

if [ ${stage} -le 2 ]; then
  for dir in "datasets/l2arctic_train" "datasets/l2arctic_test"; do
    cp $dir/cpl.txt $dir/text || exit 1 # fix_data_dir.sh expects a file named text

    utils/fix_data_dir.sh $dir || exit 1
    utils/data/get_utt2dur.sh $dir || exit 1 # get utt2dur

    cp $dir/text $dir/cpl.txt || exit 1 # cp text to cpl.txt
  done
fi

if [ ${stage} -le 3 ]; then
  ./local/data-librispeech.sh || exit 1

  # use librispeech train_all and l2arctic_train as the training data
  rm -rf datasets/train
  utils/combine_data.sh datasets/train data/train_all datasets/l2arctic_train || exit 1
fi

if [ ${stage} -le 4 ]; then
  python vq_vae/preprocess.py || exit 1
fi
