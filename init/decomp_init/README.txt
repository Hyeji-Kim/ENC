
Description
-----------

./0_gen_rmax_conf.sh ${MODEL} ${CONV}
    - generate the json file including maximum rank of each layer

./1_decomp_net.sh ${MODEL}
    - decompose the original network with maximum rank

./2_gen_NetDef_Norms.sh ${MODEL} ${CONV}
    - generate the network definition files (Net_def.txt, C_norm.txt ,...)

./3_gen_a_norm.sh ${MODEL} ${ACC_SEL}
    - generate A_norm.txt by validation dataset
    - just skip this step, if you don't want to use 'validation accuracy' as the layer-wise accuracy metric


