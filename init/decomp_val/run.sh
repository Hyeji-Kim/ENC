#export MODEL="vgg16"
export MODEL=$1 #"vgg16"
export TYPE="rmax"
export COMP="total"

python init.py \
 	--orig_deploy ${MODEL}/models_${MODEL}/deploy.prototxt \
 	--orig_train_val ${MODEL}/models_${MODEL}/train_val.prototxt \
	--orig_weights ${MODEL}/models_${MODEL}/${MODEL}.caffemodel \
 	--priv_config ${MODEL}/comp_${MODEL}/config.json.${TYPE} \
	--priv_weights ${MODEL}/comp_${MODEL}/model/${MODEL}_svd_${TYPE}.caffemodel \
	--priv_deploy ${MODEL}/comp_${MODEL}/deploy/deploy_conv_${TYPE}.prototxt \
	--priv_train_val ${MODEL}/comp_${MODEL}/train_val/train_val_conv_${TYPE}.prototxt \
	--priv_solver  ${MODEL}/comp_${MODEL}/tuning/solver.prototxt \
 	--rset_config ${MODEL}/comp_${MODEL}/config_rset.json.${TYPE} \
	--log  ${MODEL}/comp_${MODEL}/log/${TYPE}.log \
	--type ${MODEL}

