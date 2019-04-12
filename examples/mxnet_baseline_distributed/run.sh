#!/bin/bash
python /opt/incubator-mxnet/tools/launch.py -n 16 -s 8 -H hosts --launcher ssh \
"PYTHONPATH=$PYTHONPATH:/opt/incubator-mxnet/python python $PWD/cifar10_dist.py $@"
