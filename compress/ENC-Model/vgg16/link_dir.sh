current_dir=`pwd`
current_folder=`basename ${current_dir}`
export MODEL="${current_folder}"
export TYPE="rmax"
export COMP="total"
export DIR="../../../init/decomp_${1}"

#ln -s ${current_dir}/${DIR}/${MODEL}/comp_${MODEL}/model decomp_model

real_dir=`realpath ${DIR}`
ln -s ${real_dir}/${MODEL}/comp_${MODEL}/model decomp_model

