2019-03-20 Hyeji Kim (hyejikim89@kaist.ac.kr)
*********************************************


#NOTE : In this folder (init), the initial files are to be set. 

Directory 
---------
* decomp_init : decompose initial network model with maximum rank, make an eigenvalue file and network definition files. 
* decomp_test : verify the accuracy of network model with test dataset
* decomp_val  : verify the accuracy of network model with validation dataset


If you want to compress 'New Network Model', follow : 
------------------------------------------------------

For ./decomp_init
-----------------
    step 0 : prepare the basic files in ../base_models (follow ../base_models/README.txt)
    step 1 : set "MODEL" and "CAFFE_ROOT" in RUN.sh 
    step 2 : run file
        ./RUN_init.sh
    step 3 : set "MODEL", "CONV", and "ACC_SEL" in ./decomp_init/RUN_TOTAL.sh
    step 4 : move directory to "decomp_init"
    step 5 : run file (please refer decomp_init/README.txt)
        ./RUN_TOTAL.sh

For ./decomp_test and ./decomp_val
-----------------------------------
    step 0 : add the command for network evaluation in ./decomp_{test or val}/acc_loss_check.sh
        
        ex) ${CAFFE_ROOT} test -phase TEST -model $1 -weights $2 -gpu $4 -iterations ${TEST ITERATION} |tee -i $3 
            -> please change only ${CAFFE_ROOT} and ${TEST ITERATION}. 

        for decomp_test : ${TEST ITERATION} = (the number of test images)/(batch size) 
        for decomp_val  : ${TEST ITERATION} = (the number of validation images)/(batch size) 

