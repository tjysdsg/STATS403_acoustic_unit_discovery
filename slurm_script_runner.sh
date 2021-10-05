#!/bin/bash

source /home/storage15/tangjiyang/.bashrc
cd /home/storage15/tangjiyang/DAU-MD
export PYTHONPATH=$(pwd)
python "$@"
