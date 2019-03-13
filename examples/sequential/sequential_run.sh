#!/bin/bash
FAST_ROOT=$(cd ../..; pwd)
MXNET_LIBDIR=/opt/incubator-mxnet/lib
LIBFABRIC_ROOT=/opt/libfabric
# MXNET_LIBDIR=/home/pviviani/pviviani/opt/magnus/mxnet/lib
# LIBFABRIC_ROOT=/home/pviviani/pviviani/opt/magnus/libfabric
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIBDIR:$LIBFABRIC_ROOT/lib

./$@
