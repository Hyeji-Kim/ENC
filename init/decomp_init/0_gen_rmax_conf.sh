export MODEL=$1 #"vgg16"
export CONV=$2 #"1" # 1 : only conv layer / 0 : all layers

python 0_gen_conf.py \
 	--orig_prototxt ./${MODEL}/models_${MODEL}/train_val.prototxt \
 	--orig_weight ./${MODEL}/models_${MODEL}/${MODEL}.caffemodel \
    --sel_conv ${CONV} \
    --net_type ${MODEL}

