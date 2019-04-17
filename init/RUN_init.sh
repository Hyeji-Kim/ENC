
# ==============================================================
export MODEL="alex"
export CAFFE_ROOT="/home/hjkim/caffe3/" # should be absolute path
# ===============================================================

./0_set_init_dir.sh ${MODEL}

./1_set_init_link.sh ${MODEL} ${CAFFE_ROOT}


