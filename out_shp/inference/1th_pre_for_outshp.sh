#!/usr/bin/env bash

FOLDER='1536-1280'


#places=('CGDZ_8' 'CGDZ_10' 'CGHA_20' 'CGHA_21' 'CGHE_12' 'CGJC_22' 'CGJN_15' 'CGJQ_14' 'CGLY_17' 'CGMY_8' 'CGSG_18' 'CGSH_12' 'CGWH_17' 'CGZJ_21' 'CGZY_23' 'gengdi_B')
places=('A_1984' 'I_1984' 'K_1984' 'BJ10232D_SC_002' 'TR004126VI_007')

#echo "start split of \"no\""
## split large images into small chips ###
#for place in ${places[@]}
#do
#  echo "start img ${place}"
#  python -W ignore ./out_shp/inference/pre_for_outshp.py $FOLDER "${place}" "no"
#done

echo "start split of \"yes\""
python -W ignore ./out_shp/inference/pre_for_outshp.py $FOLDER "${place}" "yes"
echo "start rm"
rm -rf ./out_shp/inference/tmp
echo "finished!"
