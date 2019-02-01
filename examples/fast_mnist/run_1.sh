#!/bin/bash
FAST_ROOT=$(cd ../..; pwd)
MXNET_LIBDIR=$FAST_ROOT/3rdparty/mxnet/lib
LIBFABRIC_ROOT=/opt/libfabric
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIBDIR:$LIBFABRIC_ROOT/lib

export GAM_CARDINALITY=2
export GAM_RUN_LOCAL=$FAST_ROOT/bin/fastrun-local
export GAM_LOG_PREFIX=$PWD/logs

export GAM_NODE_0=node01
export GAM_SVC_PAP_0=6100
export GAM_SVC_MEM_0=6228
export GAM_SVC_DMN_0=6356

export GAM_NODE_1=node02
export GAM_SVC_PAP_1=6100
export GAM_SVC_MEM_1=6228
export GAM_SVC_DMN_1=6356
export GAM_RANK=$1

#valgrind ./fast_mnist
gdb ./fast_mnist
#./fast_mnist
