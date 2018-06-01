#!/bin/bash
MXNET_LIBDIR=/home/pvi/eclipse-workspace/FAST/3rdparty/mxnet/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIBDIR
./unit_test
