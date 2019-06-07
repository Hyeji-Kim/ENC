# ENC: Efficient Neural Network Compression
["Efficient Neural Network Compression"](https://arxiv.org/abs/1811.12781), CVPR 2019, by Hyeji Kim, Muhammad Umar Karim Khan, Chong-Min Kyung.  

### ENC is the fast network compression platform
* determination of compression ratio for each layer (convolutional, fully-connected)
* reduction of convolution filters and fully-connected nodes
* network decomposition based on truncated SVD


### This repository contains
* ./compress : proposed network compression methods 
* ./init : pre-processing scripts for a new network & network decomposition 
* ./base_models : caffemodel & prototxt files
* ./caffe : modified ```caffe.cpp``` for [Caffe](https://github.com/BVLC/caffe) framework


## Download the Compressed Networks
* https://github.com/Hyeji-Kim/ENC-models


## Requirements
* [Caffe](https://github.com/BVLC/caffe)
* MATLAB
* google.protobuf: ```sudo pip install google```  ```sudo pip install protobuf```
* python-tk: ```sudo apt-get install python-tk```
* scipy: ```sudo apt-get install libblas-dev```  ```sudo apt-get install liblapack-dev```  ```sudo pip install scipy```
* skimage: ```sudo apt-get install python-skimage```
* realpath: ```sudo apt-get realpath```

## Quick Start (ex. ResNet-56)

1. copy ```caffe/caffe.cpp``` to ```${your caffe home}/tools/.``` and rebuild :
```Shell
cp caffe/caffe.cpp ${your caffe home}/tools/.
make -j20 all
make -j20 pycaffe
```

2. modify the dataset path in below prototxt files : 
```Shell
./base_model/res56/train_val_test.prototxt
./base_model/res56/train_val_val.prototxt

./init/decomp_val/res56/models_res56/train_val.prototxt
./init/decomp_val/res56/comp_res56/train_val/train_val_conv_rmax.prototxt

./init/decomp_test/res56/models_res56/train_val.prototxt
./init/decomp_test/res56/comp_res56/train_val/train_val_conv_rmax.prototxt
```

3. modify the symbolic link of ```caffe```  to ```${your caffe home}/```:
```Shell
./init/decomp_val/caffe
./init/decomp_test/caffe
./compress/ENC-Inf/res56/caffe
./compress/ENC-Map/res56/caffe
```
**NOTE**: symbolic link of caffe directory should be the absolute path. (ex. ```/home/hjkim/ENC/caffe/```)

4. set the constraints in ```RUN.sh```.
```Shell
cd ./compress/ENC-Map/res56
vi RUN.sh # set variables:

TAR_COMP="{target complexity}" # which is (1-compression rate), range: 0.0 ~ 1.0, ex) 0.5
GPU_IDX="{avaiable gpu indice}" # ex) GPU_IDX="0" or GPU_IDX="0,2,3"
```

5. run the script :
```Shell
./RUN.sh
```

## For VGG-16 Compression

Download [[orig-vgg16]](https://www.amazon.com/clouddrive/share/rRU4MaQsuoyZNnMjydA4iLdyyef7KpIVDeIlP0NlE12) and [[decomp-vgg16]](https://www.amazon.com/clouddrive/share/CSDSTfsROq0OJmzSsoXlOsEFrKwB9ee9BtcCbP7F2Xs), then :
```Shell
mv vgg16.caffemodel ./base_models/vgg16/vgg16.caffemodel # original vgg16
mv vgg16_svd_1.caffemodel ./init/decomp_init/vgg16/comp_vgg16/model/vgg16_svd_1.caffemodel # decomposed vgg16
```


## Citation
```
@CONFERENCE{ENC_CVPR19,
  author={Hyeji Kim, Muhammad Umar Karim Khan, Chong-Min Kyung},
  title={Efficient Neural Network Compression},
  booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019},
}
```

## Reference

* Network decomposition : "https://github.com/chengtaipu/lowrankcnn"
* ResNet-56 for Cifar10 : "https://github.com/yihui-he/resnet-cifar10-caffe"
* VBMF : "https://sites.google.com/site/shinnkj23/downloads"

## Contact

hyejikim7833@gmail.com, Feel free to contact me. :-)
