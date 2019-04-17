
#------------------------
export MODEL="alex"
#------------------------

function basic_dir(){
    local new_dir=$1
    if test -d "${new_dir}"; then
        echo "It exists : ${new_dir}"
    else
        mkdir ${new_dir}
    fi
}

echo ">> Check the initial setting in ../../base_models"
if test -d "../../base_models/${MODEL}"; then
    if test -d "../../base_models/${MODEL}/base"; then
        echo "-- Confirm the base directory for ${MODEL}. =)"
    else
        echo "-- ERROR: There is no base directory, ../../base_models/${MODEL}/base. =("
        echo "-- Please go back to {../../base_models} and follow the README.txt"
        exit
    fi
else
    echo "-- ERROR: There is no initial directory, ../../base_models/${MODEL}. =("
    echo "-- Please go back to {../../base_models} and follow the README.txt"
    exit
fi

echo ">> Check the initial setting in ../../init/"
if test -d "../../init/decomp_test/${MODEL}"; then
    if test -d "../../init/decomp_val/${MODEL}"; then
        echo "-- Confirm the base directory for ${MODEL}. =)"
    else
        echo "-- ERROR: There is no base directory, ../../init/decomp_val/${MODEL}. =("
        echo "-- Please go back to {../../init} and follow the README.txt"
        exit
    fi
else
    echo "-- ERROR: There is no initial directory in ../../init/decomp_test/${MODEL}. =("
    echo "-- Please go back to {../../init} and follow the README.txt"
    exit
fi

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
basic_dir "${MODEL}"
basic_dir "${MODEL}/base"
basic_dir "${MODEL}/log"
basic_dir "${MODEL}/tmp_"

cp ../../base_models/${MODEL}/base/* ${MODEL}/base/.
cp ../../base_models/${MODEL}/eigenvalue.conf ${MODEL}/base/.
cd ${MODEL}
cp -a ../../init/decomp_val/caffe .
cd ..

# After this script, just copy ./vgg16/gen.py and ./vgg16/RUN.sh in the new directory.
cp vgg16/run.sh ${MODEL}
cp vgg16/gen.py ${MODEL}
cp -r vgg16/check ${MODEL}

