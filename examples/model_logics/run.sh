#!/bin/bash
FAST_ROOT=$(cd ../..; pwd)
MXNET_LIBDIR=$FAST_ROOT/3rdparty/mxnet/lib
LIBFABRIC_ROOT=/opt/libfabric
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIBDIR:$LIBFABRIC_ROOT/lib

#export GAM_HOME=$FAST_ROOT/3rdparty/gam/gam
#export GAM_INCS=$GAM_HOME/include
#export GAM_CONF=$GAM_HOME/conf/local.conf
#export GAM_LOCALHOST=localhost

#export GAM_LOG_PREFIX=$PWD/logs

export GAM_CARDINALITY=16

#if [[ $* == *-l* ]]; then
#export GAM_RUN_LOCAL=$FAST_ROOT/bin/gamrun-local_logging
#else
export GAM_RUN_LOCAL=$FAST_ROOT/bin/fastrun-local
#fi

#$GAM_RUN_LOCAL -v -n $GAM_CARDINALITY -l localhost ./mnist2D
$FAST_ROOT/bin/fastrun-mpi -H hosts -n $GAM_CARDINALITY "$@"
