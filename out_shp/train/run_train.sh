#!/usr/bin/env bash

FOLDER='512-512'

echo "start convert to coco"
### split large images into small chips ###
python -W ignore ./out_shp/train/pre_for_train.py $FOLDER

#echo "start dist_train"
#### do train ###
#export OMP_NUM_THREADS=1
#./tools/dist_train.sh out_shp/train/config.py $1 --work-dir "./out_shp/train/$FOLDER"

