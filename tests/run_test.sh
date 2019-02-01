#!/bin/bash
FAST_ROOT=$(cd ..; pwd)
MXNET_LIBDIR=$FAST_ROOT/3rdparty/mxnet/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIBDIR

export GAM_HOME=$FAST_ROOT/3rdparty/gam/gam
export GAM_INCS=$GAM_HOME/include
export GAM_CONF=$GAM_HOME/conf/local.conf
export GAM_LOCALHOST=localhost

export GAM_RUN_LOCAL=$FAST_ROOT/bin/fastrun-local

$GAM_RUN_LOCAL -v -n 2 -l $GAM_LOCALHOST ./bin/mxnet_worker_test

#$FAST_ROOT/bin/fastrun-mpi -H hosts -n 2 $PWD/bin/gam_unit_test
#$FAST_ROOT/bin/fastrun-mpi -H hosts -n 5 $PWD/bin/gff_farm
#$FAST_ROOT/bin/fastrun-mpi -H hosts -n 3 $PWD/bin/gff_all_reduce
#$FAST_ROOT/bin/fastrun-mpi -H hosts -n 3 $PWD/bin/gff_all_reduce_multi
#$FAST_ROOT/bin/fastrun-mpi -H hosts -n 3 $PWD/bin/gff_all_reduce_vector
#$FAST_ROOT/bin/fastrun-mpi -H hosts -n 2 $PWD/bin/mxnet_worker_test
