
# ==========================================================
TAR_COMP="0.252" # target complexity = (1 - compression rate), range: (0.0 ~ 1.0)
GPU_IDX="0,1,2" # avaiable gpu indice, separate it as ','
# ==========================================================

current_dir=`pwd`
current_folder=`basename ${current_dir}`
export MODEL="${current_folder}"

LOG="log/log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

python  gen.py \
    	--type ${MODEL} \
        --tar_comp ${TAR_COMP} \
        --gpu ${GPU_IDX} | tee -a $LOG


