#!/bin/bash
FAST_ROOT=$(cd ../..; pwd)
MXNET_LIBDIR=$FAST_ROOT/3rdparty/mxnet/lib
#MXNET_LIBDIR=/opt/incubator-mxnet/lib
#LIBFABRIC_ROOT=/opt/libfabric
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIBDIR:$LIBFABRIC_ROOT/lib

export GAM_CARDINALITY=2
export GAM_RUN_LOCAL=$FAST_ROOT/bin/fastrun-local

$GAM_RUN_LOCAL -v -n $GAM_CARDINALITY -l localhost $@