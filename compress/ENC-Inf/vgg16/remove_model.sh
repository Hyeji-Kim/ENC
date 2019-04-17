export MODEL=${2}
rm -rf decomp_model/${MODEL}_svd_rmax_${1}_*

#cp ${MODEL}/comp_${MODEL}/model/${MODEL}_svd_rmax_${1}_${2}.caffemodel ${MODEL}/comp_${MODEL}/model/select.caffemodel
#rm -rf ${MODEL}/comp_${MODEL}/model/${MODEL}_svd_rmax_${1}_*
#mv ${MODEL}/comp_${MODEL}/model/select.caffemodel ${MODEL}/comp_${MODEL}/model/${MODEL}_svd_rmax_${1}_${2}.caffemodel 
#
#cp ${MODEL}/comp_${MODEL}/train_val/train_val_conv_rmax_${1}_${2}.prototxt ${MODEL}/comp_${MODEL}/train_val/select.prototxt
#rm -rf ${MODEL}/comp_${MODEL}/train_val/train_val_conv_rmax_${1}_*
#mv ${MODEL}/comp_${MODEL}/train_val/select.prototxt ${MODEL}/comp_${MODEL}/train_val/train_val_conv_rmax_${1}_${2}.prototxt 

