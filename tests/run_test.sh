#!/bin/bash
MXNET_LIBDIR=$(cd ..; pwd)/3rdparty/mxnet/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIBDIR

export GAM_HOME=$(cd ..; pwd)/3rdparty/gam/gam
export GAM_INCS=$GAM_HOME/include
export GAM_RUN_LOCAL=./gamrun-local
export GAM_CONF=$GAM_HOME/conf/local.conf
export GAM_LOCALHOST=localhost

export GAM_RANK=0
export GAM_CARDINALITY=1
export GAM_LOG_PREFIX=

#$GAM_RUN_LOCAL -v -n 1 -l $GAM_LOCALHOST ./unit_test
$GAM_RUN_LOCAL -v -n 2 -l $GAM_LOCALHOST ./gam_unit_test
