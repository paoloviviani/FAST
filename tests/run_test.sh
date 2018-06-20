#!/bin/bash
MXNET_LIBDIR=$(cd ..; pwd)/3rdparty/mxnet/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIBDIR

export GAM_HOME=$(cd ..; pwd)/3rdparty/gam/gam
export GAM_INCS=$GAM_HOME/include
export GAM_RUN_LOCAL=./bin/gamrun-local
export GAM_CONF=$GAM_HOME/conf/local.conf
export GAM_LOCALHOST=localhost

export GAM_RANK=0
export GAM_LOG_PREFIX=

export GAM_CARDINALITY=1
$GAM_RUN_LOCAL -v -n 1 -l $GAM_LOCALHOST ./bin/unit_test
export GAM_CARDINALITY=2
$GAM_RUN_LOCAL -v -n 2 -l $GAM_LOCALHOST ./bin/gam_unit_test
