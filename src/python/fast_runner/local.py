"""Submission job for local jobs."""

import sys
import os
import logging
import subprocess
from subprocess import Popen
import signal
from collections import deque
from . import utils


def signal_handler(signal, frame):
    print('Termination triggered by signal {0}'.format(signal))
    killall()


def killall(inflight, exec_map):
    "terminate all inflight processes"
    while len(inflight) > 0:
        p = inflight.popleft()
        print "killing executor {0}".format(exec_map[p])
        p.kill()  # trigger server-side SIGKILL
    assert len(inflight) == 0


def submit(args, fast_bin_path):
    """Submit function of local jobs."""

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)

    max_nodes_per_host = 128
    base_pap = args.base_port
    base_mem = base_pap + max_nodes_per_host
    base_dmn = base_mem + max_nodes_per_host

    inflight = deque()
    exec_map = dict()

    if(args.num_workers > max_nodes_per_host):
        sys.exit('Error! Too many nodes')

    # parse hostname
    hostname = args.host

    # Add dependencies to env
    with open(args.config_file) as f:
        os.environ = utils.env_config(os.environ, f)

    # launch processes
    for e in range(args.num_workers):
        # compose environment
        pass_env = os.environ.copy()

        pass_env['FAST_LAUNCHER'] = args.launcher
        pass_env['FAST_SYNCDIR'] = args.syncdir

        pass_env['GAM_RANK'] = str(e)
        pass_env['GAM_CARDINALITY'] = str(args.num_workers)
        for var in args.env:
            k, v = var.split('=')
            pass_env[k] = str(v)

        for e_ in range(args.num_workers):
            pass_env['GAM_NODE_{0}'.format(e_)] = hostname
            pass_env['GAM_SVC_PAP_{0}'.format(e_)] = str(base_pap + e_)
            pass_env['GAM_SVC_MEM_{0}'.format(e_)] = str(base_mem + e_)
            pass_env['GAM_SVC_DMN_{0}'.format(e_)] = str(base_dmn + e_)

        print '> starting node = ' + str(e)

        p = subprocess.Popen(
            args.command, env=pass_env, stdout=sys.stdout, stderr=sys.stderr)

        inflight.append(p)
        exec_map[p] = e

    while len(inflight) > 0:
        p = inflight.popleft()

        if(p.poll() != None):
            ret = p.returncode

            if(ret != 0):
                print 'ERROR: executor {0} returned {1}'.format(exec_map[p], ret)
                killall(inflight, exec_map)
            else:
                print '> completed node = ' + str(exec_map[p])

        else:
            inflight.append(p)
