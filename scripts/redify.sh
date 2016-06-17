#!/bin/bash
PROTO_DIR="../caffe/src/caffe/proto/caffe.proto"
CAFFE_LY_SRC_DIR="../caffe/src/caffe/layers"
CAFFE_LY_HEADER_DIR="../caffe/include/caffe/layers"
CAFFE_SRC_DIR="../caffe/src/caffe"
CAFFE_HEADER_DIR="../caffe/include/caffe"
REDEYE_SRC="../src"

cp $REDEYE_SRC/*_layer.{cpp,cu} $CAFFE_LY_SRC_DIR
cp $REDEYE_SRC/*_layer.hpp $CAFFE_LY_HEADER_DIR
cp $REDEYE_SRC/data_transformer.cpp $CAFFE_SRC_DIR
cp $REDEYE_SRC/data_transformer.hpp $CAFFE_HEADER_DIR

i="$(grep -n message\ LayerParameter\ { $PROTO_DIR | cut -f1 -d:)"
ID_STRING="$(sed -n "$((i-1))p" $PROTO_DIR)"
AVAIL_ID="$(echo $ID_STRING | grep -o -E '[0-9]+')"
NEXT_AVAI_ID=$((AVAIL_ID+3))
sed -i "$((i-1))s/.*/\/\/\ LayerParameter\ next\ available\ layer-specific\ ID:\ $NEXT_AVAI_ID\ (last\ added:\ energy_loss_param)/" $PROTO_DIR

NOISE_PARAM_STRING=$(echo "\ \ optional NoiseParameter noise_param = $AVAIL_ID;")
ENERGY_LOSS_PARAM_STRING=$(echo "\ \ optional EnergyLossParameter energy_loss_param = $((AVAIL_ID+1));")
QUANTIZATION_PARAM_STRING=$(echo "\ \ optional QuantizationParameter quantization_param = $((AVAIL_ID+2));")

GAMMA_PARAM_STRING=$(echo "\ \ optional float gamma = 8 [default = 2.2];")

sed -i "$((i+1))i$NOISE_PARAM_STRING" $PROTO_DIR
sed -i "$((i+2))i$ENERGY_LOSS_PARAM_STRING" $PROTO_DIR
sed -i "$((i+3))i$QUANTIZATION_PARAM_STRING" $PROTO_DIR

j="$(grep -n message\ TransformationParameter\ { $PROTO_DIR | cut -f1 -d:)"
sed -i "$((j+1))i$GAMMA_PARAM_STRING" $PROTO_DIR

cat ${REDEYE_SRC}/proto_layer_info >> $PROTO_DIR
echo "Done modifying proto file."