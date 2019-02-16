#!/bin/bash
FAST_ROOT=$(cd ../..; pwd)
MXNET_LIBDIR=$FAST_ROOT/3rdparty/mxnet/lib
LIBFABRIC_ROOT=/opt/libfabric
#MXNET_LIBDIR=/home/pviviani/pviviani/opt/magnus/mxnet/lib
#LIBFABRIC_ROOT=/home/pviviani/pviviani/opt/magnus/libfabric

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIBDIR:$LIBFABRIC_ROOT/lib

export GAM_RUN_LOCAL=$FAST_ROOT/bin/fastrun-local

$FAST_ROOT/bin/fastrun-mpi -H hosts -n 12 "$PWD/$1 4 3"
#$FAST_ROOT/bin/fastrun-slurm -n 2 "$PWD/$1"