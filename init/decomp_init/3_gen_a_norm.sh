export MODEL=$1 #"vgg16"
export ACC_SEL=$2 #"1" # 1 : top-1 / 5 : top-5

# validate the accuracy for layer-wise accuracy metric
cd ../decomp_val
./run.sh ${MODEL}

# make A_norm.txt
cd ../decomp_init
python  3_a_norm.py \
		--model ${MODEL} \
        --acc_sel ${ACC_SEL}

