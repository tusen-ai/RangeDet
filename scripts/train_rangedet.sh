#!/bin/sh
DIRNAME="config/rangedet"
CONFIG="rangedet_wo_aug_4_18e"
EPOCH="18"
./scripts/horovodrun.sh 8 "python tools/train.py --config $DIRNAME/$CONFIG"
python tools/test.py --config $DIRNAME/$CONFIG
python tools/create_prediction_bin_3d.py -c $CONFIG -e $EPOCH
