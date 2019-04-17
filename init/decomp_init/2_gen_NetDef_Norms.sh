MODEL=$1 #"vgg16"
CONV=$2 #"1" # 1 : only conv layer / 0 : conv + fc layer

python 2_cal_from_conf.py \
 	--orig_prototxt ${MODEL}/models_${MODEL}/train_val.prototxt \
 	--orig_weight ${MODEL}/models_${MODEL}/${MODEL}.caffemodel \
 	--config config.json.rmax \
	--conv ${CONV} \
	--model ${MODEL}

