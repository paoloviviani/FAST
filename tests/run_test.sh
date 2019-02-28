#!/bin/bash
FAST_ROOT=$(cd ..; pwd)
MXNET_LIBDIR=$FAST_ROOT/3rdparty/mxnet/lib
#MXNET_LIBDIR=/opt/incubator-mxnet/lib
#LIBFABRIC_ROOT=/opt/libfabric
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MXNET_LIBDIR

# PARSE ARGUMENTS
show_help ()
{
  echo "Usage:

  ${0##*/} [-h][-l launcher] test_name

Options:

  -h, --help
    display this help and exit

  -l, --launcher
    localhost, mpi, slurm
  "
  echo $TESTS
}


if [[ $# -eq 0 ]]; then
	show_help
	exit
fi

EXECUTABLE=${@: -1}
ALL=false

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -l|--launcher)
    export LAUNCHER="$2"
    shift # past argument
    shift # past value
    ;;
    -h|--help)
	show_help
	exit
    ;;
    *)    # unknown option
    show_help
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    exit
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters


export GAM_HOME=$FAST_ROOT/3rdparty/gam/gam
export GAM_INCS=$GAM_HOME/include
export GAM_CONF=$GAM_HOME/conf/local.conf
export GAM_LOCALHOST=localhost

export FAST_RUN_LOCAL=$FAST_ROOT/bin/fastrun-local
export FAST_RUN_MPI=$FAST_ROOT/bin/fastrun-mpi
export FAST_RUN_SLURM=$FAST_ROOT/bin/fastrun-slurm

if [[ $LAUNCHER == localhost ]]; then
	#$FAST_RUN_LOCAL -v -n 1 -l $GAM_LOCALHOST ./bin/unit_test
	#$FAST_RUN_LOCAL -v -n 2 -l $GAM_LOCALHOST ./bin/gam_unit_test
	#$FAST_RUN_LOCAL -v -n 5 -l $GAM_LOCALHOST ./bin/gff_farm
	#$FAST_RUN_LOCAL -v -n 2 -l $GAM_LOCALHOST ./bin/gff_training_mockup
	#$FAST_RUN_LOCAL -v -n 2 -l $GAM_LOCALHOST ./bin/gff_training_concurrent
	#$FAST_RUN_LOCAL -v -n 2 -l $GAM_LOCALHOST ./bin/gff_training_concurrent_2
	#$FAST_RUN_LOCAL -v -n 9 -l $GAM_LOCALHOST ./bin/gff_training_concurrent_grid
	#$FAST_RUN_LOCAL -v -n 1 -l $GAM_LOCALHOST ./bin/mxnet_aux_test
	$FAST_RUN_LOCAL -v -n 2 -l $GAM_LOCALHOST ./bin/mxnet_worker_test
elif [[ $LAUNCHER == mpi ]]; then
	$FAST_RUN_MPI -H hosts -n 2 $PWD/bin/gam_unit_test
	$FAST_RUN_MPI -H hosts -n 5 $PWD/bin/gff_farm
	$FAST_RUN_MPI -H hosts -n 2 $PWD/bin/gff_training_mockup
	$FAST_RUN_MPI -H hosts -n 2 $PWD/bin/gff_training_concurrent
	$FAST_RUN_MPI -H hosts -n 2 $PWD/bin/gff_training_concurrent_2
	$FAST_RUN_MPI -H hosts -n 2 $PWD/bin/mxnet_aux_test
	$FAST_RUN_MPI -H hosts -n 2 $PWD/bin/mxnet_worker_test
elif [[ $LAUNCHER == slurm ]]; then
	$FAST_RUN_SLURM -H hosts -n 2 $PWD/bin/gam_unit_test
	$FAST_RUN_SLURM -H hosts -n 5 $PWD/bin/gff_farm
	$FAST_RUN_SLURM -H hosts -n 2 $PWD/bin/mxnet_worker_test
	$FAST_RUN_SLURM -H hosts -n 2 $PWD/bin/gff_training_mockup
fi