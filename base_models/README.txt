2019-03-20 Hyeji Kim (hyejikim89@kaist.ac.kr)
*********************************************

#NOTE : This folder includes the basic files of each network model such as caffemodel, prototxt, and generated network specification files. 

For each model folder, it should have below basic files:
---------------------------------------------------------
WARNING : when you determine the name of ${MODEL} such as 'res56', ${MODEL} MUST be same in all directories.

    ${MODEL}/${MODEL}.caffemodel : original baseline caffemodel
    ${MODEL}/train_val_val.prototxt : original prototxt file (TEST dataset : validation dataset)
    ${MODEL}/train_val_test.prototxt : original prototxt file (TEST dataset : test dataset)


Initial steps:
--------------
    step 0 : extract the validataion dataset 
        >> please refer ./util/extract_lmdb.py

    step 1 : prepare the basic files (*.caffemodel, *.prototxt)
        ex) res56.caffemodel, train_val.prototxt

    step 2 : make two prototxt files and change the dataset path of {phase: TRAIN, TEST}
        >> train_val_test.prototxt : {phase: TEST} -> original test dataset
        >> train_val_val.prototxt  : {phase: TEST} -> extracted validatation dataset

 
If you are done: 
----------------
>> Go to ../init directory and follow the ../init/README.txt


