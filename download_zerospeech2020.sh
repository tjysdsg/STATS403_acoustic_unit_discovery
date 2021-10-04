#!/bin/bash

PASSWORD=sM@pv7bT
for ext in zip z01 z02; do
    wget https://download.zerospeech.com/2020/zerospeech2020.$ext || exit 1
done
7z x -p$PASSWORD zerospeech2020.zip || exit 1
