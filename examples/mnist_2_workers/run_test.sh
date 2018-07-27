#!/bin/bash
FAST_ROOT=$(cd ../..; pwd)
MXNET_LIBDIR=$FAST_ROOT/3rdparty/mxnet/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIBDIR

export GAM_HOME=$FAST_ROOT/3rdparty/gam/gam
export GAM_INCS=$GAM_HOME/include
export GAM_CONF=$GAM_HOME/conf/local.conf
export GAM_LOCALHOST=localhost

export GAM_LOG_PREFIX=

# Use no more than 2
export GAM_CARDINALITY=1

if [[ $* == *-l* ]]; then
export GAM_RUN_LOCAL=$FAST_ROOT/bin/gamrun-local_logging
else
export GAM_RUN_LOCAL=$FAST_ROOT/bin/gamrun-local
fi
$GAM_RUN_LOCAL -v -n $GAM_CARDINALITY -l $GAM_LOCALHOST ./mnist_cpu
