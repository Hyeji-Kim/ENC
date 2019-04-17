export MODEL="vgg16"
export CONV="1" # 1 : only conv layer / 0 : all layers
export ACC_SEL="1" # 1 : top-1 accuracy / 5 : top-5 accuracy

# generate the json file including maximum rank of each layer
./0_gen_rmax_conf.sh ${MODEL} ${CONV}

# decompose the original network with maximum rank
./1_decomp_net.sh ${MODEL}

# generate the network definition files (Net_def.txt, C_norm.txt ,...)
./2_gen_NetDef_Norms.sh ${MODEL} ${CONV}

# generate A_norm.txt by validation dataset
./3_gen_a_norm.sh ${MODEL} ${ACC_SEL}


