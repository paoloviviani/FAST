#!/bin/bash
FAST_ROOT=$(cd ../../..; pwd)
MXNET_LIBDIR=$FAST_ROOT/3rdparty/mxnet/lib
LIBFABRIC_ROOT=/opt/libfabric
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIBDIR:$LIBFABRIC_ROOT/lib


$@
