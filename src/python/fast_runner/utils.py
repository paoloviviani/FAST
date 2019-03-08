import json
import sys
import subprocess
import os
import distutils.spawn


def which(program):
    '''
    Check if command exists
    '''
    sys_exec = distutils.spawn.find_executable(program)
    if sys_exec:
        return os.path.abspath(os.path.expanduser(sys_exec))
    else:
        return None

    
def get_mpi_env(envs):
    '''
    Get the mpirun command for setting the envornment
    support both openmpi and mpich2
    '''
    cmd = []
    # windows hack: we will use msmpi
    if sys.platform == 'win32':
        for k, v in envs.items():
            cmd.append('-env %s %s' % (k, str(v)))
        return cmd

    # decide MPI version.
    (out, err) = subprocess.Popen(['mpirun', '--version'],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE).communicate()
    if b'Open MPI' in out:
        for k, v in envs.items():
            cmd.append('-x %s=%s' % (k, str(v)))
    elif b'mpich' in err:
        for k, v in envs.items():
            cmd.append('-env %s %s' % (k, str(v)))
    else:
        raise RuntimeError('Unknown MPI Version')
    return cmd


def env_config(env, config_file):
    '''
    Returns a dictionary with the content of the JSON file 
    translated into environment variables
    '''
    temp=json.load(config_file)
    for k, v in temp.items():
        # ignore "commented" json entries
        if not '#' in k:
            if k in env:
                env[k] = env[k] + ':' + v
            else:
                env[k] = v
    return env

def scriptinfo():
    '''
    Returns a dictionary with information about the running top level Python
    script:
    ---------------------------------------------------------------------------
    dir:    directory containing script or compiled executable
    name:   name of script or executable
    source: name of source code file
    ---------------------------------------------------------------------------
    "name" and "source" are identical if and only if running interpreted code.
    When running code compiled by py2exe or cx_freeze, "source" contains
    the name of the originating Python script.
    If compiled by PyInstaller, "source" contains no meaningful information.
    '''

    import os, sys, inspect
    #---------------------------------------------------------------------------
    # scan through call stack for caller information
    #---------------------------------------------------------------------------
    for teil in inspect.stack():
        # skip system calls
        if teil[1].startswith("<"):
            continue
        if teil[1].upper().startswith(sys.exec_prefix.upper()):
            continue
        trc = teil[1]
        
    # trc contains highest level calling script name
    # check if we have been compiled
    if getattr(sys, 'frozen', False):
        scriptdir, scriptname = os.path.split(sys.executable)
        return {"dir": scriptdir,
                "name": scriptname,
                "source": trc}

    # from here on, we are in the interpreted case
    scriptdir, trc = os.path.split(trc)
    # if trc did not contain directory information,
    # the current working directory is what we need
    if not scriptdir:
        scriptdir = os.getcwd()

    scr_dict ={"name": trc,
               "source": trc,
               "dir": scriptdir}
    return scr_dict