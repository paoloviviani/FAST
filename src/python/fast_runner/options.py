"""Command line options of job submission script."""
import os
import argparse
import datetime
from . import utils


def none_or_str(value):
      if value == 'none' or value == 'None':
            return None
      return value


def get_opts(args=None):
      """Get options to launch the job.
      Returns
      -------
      args: ArgumentParser.Argument
      The arguments returned by the parser.
      cache_file_set: set of str
      The set of files to be cached to local execution environment.
      """
      timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

      parser = argparse.ArgumentParser(description='FAST job submission.')
      parser.add_argument('-l', '--launcher', type=str, default="local",
                  choices=['slurm', 'mpi', 'local', 'ssh'],
                  help=('Cluster type of this submission,' +
                        'default to local'))
      parser.add_argument('-n', '--num-workers', required=True, type=int,
                  help='Number of worker proccess to be launched.')
      parser.add_argument('--queue', default=None, type=str,
                  help='SLURM only. The submission queue the job should goes to.')
      parser.add_argument('-p', '--ssh-port', default=None, type=str,
                  help='SLURM only. The submission queue the job should goes to.')
      parser.add_argument('--log-file', default=None, type=str,
                  help=('Output log to the specific log file'))
      parser.add_argument('--base-port', default=6000, type=int,
                  help=('Starting port for libfabric-based communication, ' +
                        'at least 256 consecutive ports should be free.'))
      parser.add_argument('--log-level', default='INFO', type=str,
                  choices=['INFO', 'DEBUG'],
                  help='Logging level of the logger.')
      parser.add_argument('--host', default='localhost', type=str,
                  help=('Local address if not localhost'))
      parser.add_argument('--syncdir', default=os.path.join(os.getcwd(), str('fastjob-'+timestamp)), type=str,
                  help=('Directory used for synchronization and storage of results'))
      parser.add_argument('--config-file', default=os.path.join(utils.scriptinfo()['dir'], 'config.json'), type=str,
                  help=('JSON config file for dependencies and global env '))
      parser.add_argument('-H', '--machinefile', '--hostfile', default=None, type=str,
                  help=('The file contains the list of hostnames, needed for MPI and ssh.'))
      parser.add_argument('--env', action='append', default=[],
                  help='Node code environment variables.')
      parser.add_argument('command', nargs='+',
                  help='Command to be launched')

      (args, unknown) = parser.parse_known_args(args)
      args.command += unknown

      return args
