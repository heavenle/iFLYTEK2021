#!/bin/bash

FOLDER='1536-1280'
# FOLDER=$1
## places=(\
##  'CGDZ_9' 'CGGZ_7' 'CGHA_5' 'CGHE_2' 'CGHE_13' 'CGJC_23' \
##  'CGJC_24' 'CGJN_2' 'CGJQ_3' 'CGLF_9' 'CGLY_4' 'CGTL_25' \
##  'CGWH_4' 'CGWH_18' 'CGXC_3' 'CGZJ_6' 'CGZY_7' 'CGZY_24')

places=('CGDZ_8' 'CGDZ_10' 'CGHA_20' 'CGHA_21' 'CGHE_12' 'CGJC_22' 'CGJN_15' 'CGJQ_14' 'CGLY_17' 'CGMY_8' 'CGSG_18' 'CGSH_12' 'CGWH_17' 'CGZJ_21' 'CGZY_23' 'gengdi_B')

### split large images into small chips ###
#for place in ${places[@]}
#do
#  python -W ignore ./out_shp/inference/pre_for_outshp.py $FOLDER "${place}" "no" &
#done

#python -W ignore ./out_shp/inference/pre_for_outshp.py $FOLDER "${place}" "yes"
#rm -rf tmp


### do inference ###
#export OMP_NUM_THREADS=1
#./tools/dist_test.sh out_shp/inference/htc_gengdi/config.py out_shp/inference/htc_gengdi/epoch_31.pth $1 --format-only --eval-options "jsonfile_prefix=./out_shp/inference/$FOLDER"
# ./tools/dist_test.sh out_shp/inference/htc-800/config.py out_shp/inference/htc-800/epoch_9.pth $1 --format-only --eval-options "jsonfile_prefix=./out_shp/inference/$FOLDER"

### merge outputs and generate submission file ###
for place in ${places[@]}
do
  python -W ignore ./out_shp/inference/single_shp_out.py ${FOLDER} "${place}" &
done

# wait for all processes done
num=$(ls -l out_shp/inference/${FOLDER}/out_shp | grep "^-" | wc -l)
while [ $num -lt 72 ]; do
  sleep 1;
  num=$(ls -l out_shp/inference/${FOLDER}/out_shp/ | grep "^-" | wc -l);
done

sleep 3;

cd out_shp/inference/${FOLDER}/out_shp
zip submit_${FOLDER}.zip ./*
mv submit_${FOLDER}.zip ../
cd ....
