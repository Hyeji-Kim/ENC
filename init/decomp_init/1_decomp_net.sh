export MODEL=$1 #"vgg16"
export TYPE="rmax"

python 1_decomp.py \
    --model ${MODEL}/models_${MODEL}/deploy.prototxt \
    --train ${MODEL}/models_${MODEL}/train_val.prototxt \
    --config ${MODEL}/comp_${MODEL}/config.json.${TYPE} \
    --weights ${MODEL}/models_${MODEL}/${MODEL}.caffemodel \
    --save_weights ${MODEL}/comp_${MODEL}/model/${MODEL}_svd.caffemodel \
    --save_model ${MODEL}/comp_${MODEL}/deploy/deploy_conv.prototxt \
    --save_train ${MODEL}/comp_${MODEL}/train_val/train_val_conv.prototxt \
    --net_type ${MODEL}

