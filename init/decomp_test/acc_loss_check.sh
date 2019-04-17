
if [ "$5" = "vgg16" ]; then
    caffe/build/tools/caffe test -phase TEST -model $1 -weights $2 -gpu $4 -iterations 500 |tee -i $3 # VGG16 (ImageNet)

elif [ "$5" = "res56" ]; then
    caffe/build/tools/caffe test -phase TEST -model $1 -weights $2 -gpu $4 -iterations 100 |tee -i $3 # ResNet-56 (Cifar10)

else
    echo "else"
fi


