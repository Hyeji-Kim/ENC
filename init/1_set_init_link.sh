
# ===========================================
export MODEL=$1 
export CAFFE_ROOT=$2 #"/home/hjkim/caffe3/" # absolute path
# ===========================================


function link_gen(){
    local FILENAME=$2
    local LINK_PATH=$1
    if [ -e "$FILENAME" ]; then
        echo "-- link exist : ${FILENAME}"
    else
        echo "-- link not exist : ${FILENAME}"
        ln -s $LINK_PATH $FILENAME
    fi
}

function basic_path(){
    link_gen "${CAFFE_ROOT}" "caffe"
    cd ${MODEL}
    real_dir=`realpath ../../../`
    link_gen "${real_dir}/base_models/${MODEL}/eigenvalue.conf" "eigenvalue.conf"

    cd models_${MODEL}
    real_dir=`realpath ../../../../`
    link_gen "${real_dir}/base_models/${MODEL}/${MODEL}.caffemodel" "${MODEL}.caffemodel"
    link_gen "${real_dir}/base_models/${MODEL}/train_val_${1}.prototxt" "train_val.prototxt"
    link_gen "${real_dir}/base_models/${MODEL}/deploy.prototxt" "deploy.prototxt"

    cd ../comp_${MODEL}
    link_gen "${real_ir}/base_models/${MODEL}/config.json.rmax" "config.json.rmax"
}

function decomp_path(){
    cd model
    real_dir=`realpath ../../../../`
    link_gen "${real_dir}/decomp_init/${MODEL}/comp_${MODEL}/model/${MODEL}_svd_1.caffemodel" "${MODEL}_svd_rmax.caffemodel"

    cd ../train_val
    link_gen "${real_dir}/decomp_init/${MODEL}/comp_${MODEL}/train_val/train_val_conv_1.prototxt" "train_val_conv_rmax.prototxt"

    cd ../tuning
    link_gen "${real_dir}/base_models/${MODEL}/solver.prototxt" "solver.prototxt"
}

#INIT_PATH=$( cd "$(dirname "$0")" ; pwd )
INIT_PATH=`pwd`
echo "CURRENT PATH - decomp_init"
cd decomp_init
if test -d "${MODEL}"; then
    basic_path "test"
    echo "-- done"
else
    echo "-- there is no directory : ./decomp_init/${MODEL}"
fi

cd ${INIT_PATH}
echo "CURRENT PATH - decomp_val"
cd decomp_val
if test -d "${MODEL}"; then
    basic_path "val"
    real_dir=`realpath ../../../../`
    link_gen "${real_dir}/base_models/${MODEL}/base/Rset.txt" "${MODEL}.txt"
    decomp_path
    echo "-- done"
else
    echo "-- there is no directory : ./decomp_val/${MODEL}"
fi

cd ${INIT_PATH}
echo "CURRENT PATH - decomp_test"
cd decomp_test
if test -d "${MODEL}"; then
    basic_path "test"
    >${MODEL}.txt
    decomp_path
    echo "-- done"
else
    echo "-- there is no directory : ./decomp_test/${MODEL}"
fi

