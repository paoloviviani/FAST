#!/bin/bash
FAST_ROOT=$(cd ..; pwd)
MXNET_LIBDIR=$FAST_ROOT/3rdparty/mxnet/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIBDIR

export GAM_HOME=$FAST_ROOT/3rdparty/gam/gam
export GAM_INCS=$GAM_HOME/include
export GAM_CONF=$GAM_HOME/conf/local.conf
export GAM_LOCALHOST=localhost

export GAM_CARDINALITY=2

if [[ $* == *-l* ]]; then
export GAM_RUN_LOCAL=$FAST_ROOT/bin/gamrun-local_logging
else
export GAM_RUN_LOCAL=$FAST_ROOT/bin/gamrun-local
fi

#$GAM_RUN_LOCAL -v -n 1 -l $GAM_LOCALHOST ./bin/unit_test
#$GAM_RUN_LOCAL -v -n 2 -l $GAM_LOCALHOST ./bin/gam_unit_test
$GAM_RUN_LOCAL -v -n 6 -l $GAM_LOCALHOST ./bin/gff_farm
$GAM_RUN_LOCAL -v -n 4 -l $GAM_LOCALHOST ./bin/gff_all_reduce