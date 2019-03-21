"""Submission job for mpi jobs."""

import sys
import os
import subprocess
import logging
import signal
from . import utils

def submit(args, fast_bin_path):

    max_nodes_per_host = 128
    base_pap = args.base_port
    base_mem = base_pap + max_nodes_per_host
    base_dmn = base_mem + max_nodes_per_host

    if(args.num_workers > max_nodes_per_host):
        sys.exit('Error! Too many nodes')

    pass_env = os.environ.copy()
    pass_env['GAM_CARDINALITY'] = str(args.num_workers)
    pass_env['FAST_LAUNCHER'] = args.launcher
    pass_env['FAST_SYNCDIR'] = args.syncdir
    
    if not os.environ.get('LD_LIBRARY_PATH') == None:
        pass_env['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH')
    if not os.environ.get('PATH') == None:
        pass_env['PATH'] = os.environ.get('PATH')
    for var in args.env:
        k, v = var.split('=')
        pass_env[k] = str(v)
    with open(args.config_file) as f:
        pass_env = utils.env_config(pass_env, f)

    hosts = []
    hosts_string = os.popen('scontrol show hostnames $SLURM_NODELIST').read()
    hosts = hosts_string.split()

    if len(hosts) < args.num_workers:
        sys.exit('Error! Not enough nodes')

    for e_ in range(args.num_workers):
        pass_env['GAM_NODE_{0}'.format(e_)] = hosts[e_]
        pass_env['GAM_SVC_PAP_{0}'.format(e_)] = str(base_pap)
        pass_env['GAM_SVC_MEM_{0}'.format(e_)] = str(base_mem)
        pass_env['GAM_SVC_DMN_{0}'.format(e_)] = str(base_dmn)

    cmd_list = []
    cmd_list.append('srun')
    cmd_list.append('--export=ALL')
    cmd_list.append('-N' + str(args.num_workers))
    cmd_list.append('--ntasks-per-node=1')
    cmd_list.append('-l')
    cmd_list.extend(args.command)
    cmd_string = ' '.join(cmd_list)
    
    ret = subprocess.call(
        cmd_string, env=pass_env, stdout=sys.stdout, stderr=sys.stderr, shell=True)

    if(ret != 0):
        print 'SLURM ERROR: srun returned error ' + str(ret)

    killall()