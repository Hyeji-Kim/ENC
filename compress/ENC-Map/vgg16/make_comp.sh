current_dir=`pwd`
current_folder=`basename ${current_dir}`
export MODEL="${current_folder}"
export TYPE="rmax"
export COMP="total"
export DIR="../../../init/decomp_${2}"


cp ${current_dir}/${DIR}/acc_loss_check.sh .

rm -rf stage1_${MODEL}_${1}.txt
cp check/init/gpu${1}.txt check/.

python comp_model.py \
    --orig_deploy ${current_dir}/${DIR}/${MODEL}/models_${MODEL}/deploy.prototxt \
    --orig_train_val ${current_dir}/${DIR}/${MODEL}/models_${MODEL}/train_val.prototxt \
    --orig_weights ${current_dir}/${DIR}/${MODEL}/models_${MODEL}/${MODEL}.caffemodel \
    --priv_config ${current_dir}/${DIR}/${MODEL}/comp_${MODEL}/config.json.${TYPE} \
    --priv_weights ${current_dir}/${DIR}/${MODEL}/comp_${MODEL}/model/${MODEL}_svd_${TYPE}.caffemodel \
    --priv_deploy ${current_dir}/${DIR}/${MODEL}/comp_${MODEL}/deploy/deploy_conv_${TYPE}.prototxt \
    --priv_train_val ${current_dir}/${DIR}/${MODEL}/comp_${MODEL}/train_val/train_val_conv_${TYPE}.prototxt \
    --priv_solver ${current_dir}/${DIR}/${MODEL}/comp_${MODEL}/tuning/solver.prototxt \
    --log  ${current_dir}/${DIR}/${MODEL}/comp_${MODEL}/log/${TYPE}.log \
    --type ${MODEL} \
    --txt_idx $1 \
    --acc_type $2

cp check/end/gpu${1}.txt check/.
mv stage* tmp_
