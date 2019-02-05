#!/bin/bash
FAST_ROOT=$(cd ../..; pwd)
MXNET_LIBDIR=$FAST_ROOT/3rdparty/mxnet/lib
LIBFABRIC_ROOT=/opt/libfabric
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIBDIR:$LIBFABRIC_ROOT/lib

export GAM_CARDINALITY=16
export GAM_RUN_LOCAL=$FAST_ROOT/bin/fastrun-local

$FAST_ROOT/bin/fastrun-mpi -H hosts -n $GAM_CARDINALITY "$PWD/$1 4 4"
