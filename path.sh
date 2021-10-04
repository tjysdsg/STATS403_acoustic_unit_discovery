#!/bin/bash
# KALDI_ROOT=/home/storage04/zhuangweiji/git/kaldi-master-cuda10.1/  # compute-gop is not the newest
KALDI_ROOT=/home/storage06/wangyongqing/work/kaldi-cuda11.1/kaldi/
if [ ! -d $KALDI_ROOT ]; then
  echo KALDI_ROOT has not been set.
  exit
fi

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LD_LIBRARY_PATH=$KALDI_ROOT/tools/openfst/lib:$KALDI_ROOT/tools/liblbfgs-1.10/lib
export LC_ALL=C

export CUDA_CACHE_DISABLE=1

# SRILM is needed for LM model building
SRILM_ROOT=$KALDI_ROOT/tools/srilm
SRILM_PATH=$SRILM_ROOT/bin:$SRILM_ROOT/bin/i686-m64
export PATH=$PATH:$SRILM_PATH

