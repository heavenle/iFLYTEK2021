#!/usr/bin/env bash

FOLDER='1536-1280'

python tools/test.py out_shp/inference/htc_gengdi_80epoch/config.py out_shp/inference/htc_gengdi_80epoch/epoch_80.pth --format-only --eval-options "jsonfile_prefix=./out_shp/inference/$FOLDER"
