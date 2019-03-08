"""Submission job for mpi jobs."""

import sys
import os
import subprocess
import logging
import signal
from . import utils

def submit(args, fast_bin_path):
    def killall():
        print "Cleaning up MPI processes"
        cleanup =  ['mpirun', '--pernode', '--hostfile', args.machinefile, 'orte-clean', '--verbose']
        ret = subprocess.call(
            cleanup, env=pass_env, stdout=sys.stdout, stderr=sys.stderr)

    def signal_handler(signal, frame):
        print('Termination triggered by signal {0}'.format(signal))
        killall()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)

    max_nodes_per_host = 128
    base_pap = args.base_port
    base_mem = base_pap + max_nodes_per_host
    base_dmn = base_mem + max_nodes_per_host

    if(args.num_workers > max_nodes_per_host):
        sys.exit('Error! Too many nodes')

    pass_env = {}
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
    if args.machinefile != None and os.path.isfile(args.machinefile):
        with open(args.machinefile) as fp:
            for line in fp:
                hosts.append(line.rstrip())
    else:
        sys.exit('ERROR: host file not provided')

    if len(hosts) != args.num_workers:
        sys.exit('Error! Number of workers not equal to nodes')

    for e_ in range(args.num_workers):
        pass_env['GAM_NODE_{0}'.format(e_)] = hosts[e_]
        pass_env['GAM_SVC_PAP_{0}'.format(e_)] = str(base_pap + e_)
        pass_env['GAM_SVC_MEM_{0}'.format(e_)] = str(base_mem + e_)
        pass_env['GAM_SVC_DMN_{0}'.format(e_)] = str(base_dmn + e_)

    cmd_list = []
    cmd_list.append('mpirun')
    cmd_list.append('-n ' + str(args.num_workers))
    cmd_list.append('--pernode')
    cmd_list.append('--tag-output')
    cmd_list.append('--machinefile ' + str(args.machinefile))
    cmd_list.extend(utils.get_mpi_env(pass_env))
    cmd_list.extend(args.command)
    cmd_string = ' '.join(cmd_list)
    
    ret = subprocess.call(
        cmd_string, env=pass_env, stdout=sys.stdout, stderr=sys.stderr, shell=True)

    if(ret != 0):
        print 'MPI ERROR: mpirun returned error ' + str(ret)

    killall()