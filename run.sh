#!usr/local/bin bash

IMGS="imgs/"
LIST="list/"
MTB="mtb/"
HDR="hdr/"
EXPO="exposure/"
SRC="src/"
img=$1
key=$2
New="${img}_${key}_0.0001"

ls ${IMGS}${img} | grep .JPG > ${LIST}${img}.txt
echo "start MTB"
mkdir ${MTB}${img}
python3 ${SRC}mtb.py ${IMGS}${img} ${LIST}${img}.txt 3 ${MTB}${img}
echo "Get exposure value"
python3 ${SRC}findexp.py ${IMGS}${img} ${LIST}${img}.txt > ${EXPO}${img}.txt
echo "Imgs to HDR"
python3 ${SRC}imgsTohdr.py ${MTB}${img} ${EXPO}${img}.txt ${HDR}${New}
echo "save ${New}.hdr under ${HDR}"
echo "ToneMapping"
python3 ${SRC}photographic_tonemapping.py ${HDR}${New}.hdr $key 0.0001
