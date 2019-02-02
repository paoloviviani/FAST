#!/bin/bash
if [ ! -d "./cifar10" ]; then
  mkdir mnist_data
  (cd cifar10; wget http://data.mxnet.io/data/cifar10/cifar10_train.lst)
  (cd cifar10; wget http://data.mxnet.io/data/cifar10/cifar10_train.rec)
  (cd cifar10; wget http://data.mxnet.io/data/cifar10/cifar10_val.lst)
  (cd cifar10; wget http://data.mxnet.io/data/cifar10/cifar10_val.rec)
fi
echo "Data downloaded"