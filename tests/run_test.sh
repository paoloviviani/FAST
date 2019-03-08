#!/bin/bash
# PARSE ARGUMENTS
show_help ()
{
  echo "Usage:

  ${0##*/} launcher [local|mpi|slurm|ssh] hostfile [not needed for local]

  "
  echo $TESTS
}

if [[ $# -eq 0 ]]; then
	show_help
	exit
fi

LAUNCHER=$1
HOSTFILE=$2

FAST_ROOT=$(cd ../bin; pwd)
export PATH=:$FAST_ROOT:$PATH

if [[ $LAUNCHER == local ]]; then
	# fast-submit -n 1 -l local --node-driver none ./bin/unit_test
	# fast-submit -n 2 -l local --node-driver none ./bin/gam_unit_test
  # fast-submit -n 2 -l local --node-driver none ./bin/gam_alloc_test
	# fast-submit -n 5 -l local --node-driver none ./bin/gff_farm
	# fast-submit -n 2 -l local --node-driver none ./bin/gff_training_mockup
	# fast-submit -n 2 -l local --node-driver none ./bin/gff_training_concurrent
	# fast-submit -n 2 -l local --node-driver none ./bin/gff_training_concurrent_2
	# fast-submit -n 9 -l local --node-driver none ./bin/gff_training_concurrent_grid
	# fast-submit -n 1 -l local --node-driver none ./bin/mxnet_aux_test
	fast-submit -n 2 -l local --node-driver none ./bin/mxnet_worker_test
elif [[ $LAUNCHER == mpi ]]; then
	fast-submit -n 1 -H hosts -l mpi ./bin/unit_test
	fast-submit -n 2 -H hosts -l mpi ./bin/gam_unit_test
  fast-submit -n 2 -H hosts -l mpi ./bin/gam_alloc_test
	fast-submit -n 5 -H hosts -l mpi ./bin/gff_farm
	fast-submit -n 2 -H hosts -l mpi ./bin/gff_training_mockup
	fast-submit -n 2 -H hosts -l mpi ./bin/gff_training_concurrent
	fast-submit -n 2 -H hosts -l mpi ./bin/gff_training_concurrent_2
	fast-submit -n 9 -H hosts -l mpi ./bin/gff_training_concurrent_grid
	fast-submit -n 1 -H hosts -l mpi ./bin/mxnet_aux_test
	fast-submit -n 2 -H hosts -l mpi ./bin/mxnet_worker_test
fi