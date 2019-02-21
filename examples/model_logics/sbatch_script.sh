#!/bin/bash --login
#SBATCH --job-name=job-mxnet
#SBATCH --partition=workq
#SBATCH --nodes=100
#SBATCH --time=2:00:00
#SBATCH --account=director2044
#SBATCH --export=NONE

source /home/pviviani/magnus_env.sh

cat << EOT >> run_mxnet.conf
0      bash -c './sched_serv_work.sh'
1-9    bash -c './serv_work.sh'
10-99  bash -c './worker.sh'
EOT

srun --export=ALL -n100 --ntasks-per-node=1 -l --multi-prog run_mxnet.conf
rm run_mxnet.conf
