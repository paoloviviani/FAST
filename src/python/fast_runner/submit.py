"""Job submission script"""

import logging
import os
from . import options
from . import local
from . import mpi
from . import slurm
from . import ssh
from . import utils

def config_logger(args):
    """Configure the logger according to the arguments
    Parameters
    ----------
    args: argparser.Arguments
       The arguments passed in by the user.
    """
    fmt = '%(asctime)s %(levelname)s %(message)s'
    if args.log_level == 'INFO':
        level = logging.INFO
    elif args.log_level == 'DEBUG':
        level = logging.DEBUG
    else:
        raise RuntimeError("Unknown logging level %s" % args.log_level)

    if args.log_file is None:
        logging.basicConfig(format=fmt, level=level)
    else:
        logging.basicConfig(format=fmt, level=level, filename=args.log_file)
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(fmt))
        console.setLevel(level)
        logging.getLogger('').addHandler(console)

def main(fast_bin_path):
    args = options.get_opts()
    config_logger(args)

    try:
        os.mkdir(args.syncdir)
    except OSError:  
        print ("Creation of the synchronization directory %s failed" % args.syncdir)
    else:  
        print ("Successfully created the synchronization directory %s " % args.syncdir)

    if utils.which(args.command[0]) != None:
        args.command[0] = utils.which(args.command[0])
    else:
        sys.exit('Error! Executable not found')
    args.command.insert(0, os.path.join(fast_bin_path, 'node-driver'))

    if args.machinefile != None:
        args.machinefile = os.path.abspath(args.machinefile)
    if args.syncdir != None: 
        os.chdir(args.syncdir)

    if args.launcher == 'local':
        local.submit(args, fast_bin_path)
    elif args.launcher == 'mpi':
        mpi.submit(args, fast_bin_path)
    elif args.launcher == 'slurm':
        slurm.submit(args, fast_bin_path)
    elif args.launcher == 'ssh':
        ssh.submit(args, fast_bin_path)
    else:
        raise RuntimeError('Unknown submission cluster type %s' % args.cluster)