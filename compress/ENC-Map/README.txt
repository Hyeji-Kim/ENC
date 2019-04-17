2019-03-20 Hyeji Kim (hyejikim89@kaist.ac.kr)
*********************************************


For a new network model:
------------------------------
step0 : WARNING(!!) The initial setting in {../../init} and {../../base_models} MUST be done before the network compression.

step1 : modify the script in {./0_basic_setting.sh}
    export MODEL="${namek of network model}"     ex) export MODEL="alex"

step2 : make the initial directory
    ./0_basic_setting.sh


For the network compression:
----------------------------
step0 : WARNING(!!) The initial setting in {../../init} and {../../base_models} MUST be done before the network compression.

step1 : move to the directory to be compressed      ex) cd ./res56

step2 : modify the initial parameters in {gen.py} -> line number 38-55
        -- for res56 and vgg16, the initial parameters are already set. 

step3 : run the script
    ./RUN.sh      


