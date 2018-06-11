#!/bin/bash
MXNET_LIBDIR=$(cd ..; pwd)/3rdparty/mxnet/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIBDIR

export GAM_HOME=$(cd ..; pwd)/3rdparty/gam/gam
export GAM_INCS=$GAM_HOME/include
export GAM_RUN=$GAM_HOME/bin/gamrun
export GAM_RUN_LOCAL=$GAM_HOME/bin/gamrun-local
export GAM_CONF=$GAM_HOME/conf/local.conf
export GAM_LOCALHOST=localhost

export GAM_RANK=0
export GAM_CARDINALITY=1
export GAM_LOG_PREFIX=gam_log_

./unit_test
