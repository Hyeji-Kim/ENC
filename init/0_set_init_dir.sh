
export MODEL=$1

function basic_dir(){
    mkdir ${decomp_path}/${MODEL}
    mkdir ${decomp_path}/${MODEL}/models_${MODEL}
    mkdir ${decomp_path}/${MODEL}/comp_${MODEL}
    mkdir ${decomp_path}/${MODEL}/comp_${MODEL}/model
    mkdir ${decomp_path}/${MODEL}/comp_${MODEL}/train_val
    mkdir ${decomp_path}/${MODEL}/comp_${MODEL}/tuning
    mkdir ${decomp_path}/${MODEL}/comp_${MODEL}/deploy
    mkdir ${decomp_path}/${MODEL}/comp_${MODEL}/log
    mkdir ${decomp_path}/${MODEL}/comp_${MODEL}/log/final
}

export decomp_path="decomp_init"
basic_dir

export decomp_path="decomp_val"
basic_dir

export decomp_path="decomp_test"
basic_dir


