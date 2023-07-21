import copy
import numpy as np
import os
import sys
import shutil
from glob import glob
import pdb
import time

from dlnpyutils import utils as dln
from . import apload
from astropy.io import fits
from astropy.table import Table

import subprocess
#from slurm import queue as pbsqueue
#from slurm.models import Job,Member
from os import getlogin, getuid, getgid, makedirs, chdir
from pwd import getpwuid
from grp import getgrgid
from datetime import datetime,timezone
import string
import random
import traceback

SLURMDIR = '/scratch/general/nfs1/'

def genkey(n=20):
    characters = string.ascii_lowercase + string.digits
    key =  ''.join(random.choice(characters) for i in range(n))
    return key

def slurmstatus(label,jobid,username=None):
    """
    Get status of the job from the slurm queue
    """

    if username is None:
        username = getpwuid(getuid())[0]
    slurmdir = SLURMDIR+username+'/slurm/'
    
    # Check if the slurm job is still running
    #  this will throw an exception if the job finished already
    #  slurm_load_jobs error: Invalid job id specified
    try:
        # previously long jobnames got truncated, make sure to get the full name
        form = 'jobid,jobname%30,partition%30,account,alloccpus,state,exitcode,nodelist'
        res = subprocess.check_output(['sacct','-u',username,'-j',str(jobid),'--format='+form])
    except:
        print('Failed to get Slurm status')
        #traceback.print_exc()
        return []
    if type(res)==bytes: res=res.decode()
    lines = res.split('\n')
    # remove first two lines
    lines = lines[2:]
    # only keep lines with the label in it
    lines = [l for l in lines if ((l.find(label)>-1) and (l.find(str(jobid))>-1))]
    # JobID    JobName  Partition    Account  AllocCPUS      State ExitCode
    out = np.zeros(len(lines),dtype=np.dtype([('JobID',str,30),('JobName',str,30),('Partition',str,30),
                                              ('Account',str,30),('AllocCPUS',int),('State',str,30),
                                              ('ExitCode',str,20),('Nodelist',str,30),('done',bool)]))
    for i in range(len(lines)):
        dum = lines[i].split()
        out['JobID'][i] = dum[0]
        out['JobName'][i] = dum[1]
        out['Partition'][i] = dum[2]
        out['Account'][i] = dum[3]
        out['AllocCPUS'][i] = int(dum[4])
        out['State'][i] = dum[5]
        out['ExitCode'][i] = dum[6]
        out['Nodelist'][i] = dum[7]        
        if dum[5] == 'COMPLETED':
            out['done'][i] = True

    return out

def taskstatus(label,key,username=None):
    """
    Return tasks that completed
    """

    if username is None:
        username = getpwuid(getuid())[0]
    slurmdir = SLURMDIR+username+'/slurm/'
    jobdir = slurmdir+label+'/'+key+'/'
    
    # Get number of tasks
    ntasks = dln.readlines(jobdir+label+'.ntasks')
    ntasks = int(ntasks[0])
    
    # Load the task inventory
    #tasks = Table.read(jobdir+label+'_inventory.txt',names=['task','node','proc'],format='ascii')

    # Check the log files
    outfiles = glob(jobdir+'*'+label+'_*.out*')
    # Load the files
    tasknum = 0
    completedtasks = []
    for i in range(len(outfiles)):
        lines = dln.readlines(outfiles[i])
        # Get the "completed" lines
        clines = [l for l in lines if l.find('ompleted')>-1]
        for j in range(len(clines)):
            # Task 127 node01 proc64 Completed Sun Oct 16 19:30:01 MDT 2022
            dum = clines[j].split()
            taskid = dum[1]
            nodeid = dum[2][4:]
            procid = dum[3][4:]
            stat = [taskid,nodeid,procid]
            completedtasks += [stat]
            tasknum += 1

    if len(completedtasks):
        tstatus = np.zeros(len(completedtasks),dtype=np.dtype([('task',int),('node',int),('proc',int),('done',bool)]))
        for i,stat in enumerate(completedtasks):
            tstatus['task'][i] = stat[0]
            tstatus['node'][i] = stat[1]   # node01
            tstatus['proc'][i] = stat[2]   # proc54
            tstatus['done'][i] = True
    else:
        tstatus = []
    
    return tstatus
    
def status(label,key,jobid,username=None):
    """
    Return the status of a job.
    """

    if username is None:
        username = getpwuid(getuid())[0]
    slurmdir = SLURMDIR+username+'/slurm/'
    jobdir = slurmdir+label+'/'+key+'/'
    
    # Check if the slurm job is still running
    state = slurmstatus(label,jobid)
    if len(state)==0:
        return None,None,None
    node = len(state)
    ndone = np.sum(state['done'])
    noderunning = node-ndone
    # Get number of tasks
    ntasks = dln.readlines(jobdir+label+'.ntasks')
    ntasks = int(ntasks[0])
    # Check how many tasks have completed
    tstatus = taskstatus(label,key)
    ncomplete = len(tstatus)
    taskrunning = ntasks-ncomplete
    percent = 100*ncomplete/ntasks
    return noderunning,taskrunning,percent
    
def queue_wait(label,key,jobid,sleeptime=60,logger=None,verbose=True):
    """
    Wait until the job is done
    """

    if logger is None:
        logger = dln.basiclogger()

    username = getpwuid(getuid())[0]
    slurmdir = SLURMDIR+username+'/slurm/'
    jobdir = slurmdir+label+'/'+key+'/'

    # Get number of tasks
    ntasks = dln.readlines(jobdir+label+'.ntasks')
    ntasks = int(ntasks[0])
    
    # While loop
    done = False
    count = 0
    while (done==False):
        time.sleep(sleeptime)
        # Check that state
        noderunning,taskrunning,percent = status(label,key,jobid)
        if noderunning is not None:
            # Check if the slurm job is still running
            state = slurmstatus(label,jobid)
            node = len(state)
            ndone = np.sum(state['done'])
            noderunning = node-ndone
            # Get number of tasks
            ntasks = dln.readlines(jobdir+label+'.ntasks')
            ntasks = int(ntasks[0])
            # Check how many tasks have completed
            tstatus = taskstatus(label,key)
            ncomplete = len(tstatus)
            taskrunning = ntasks-ncomplete
            percent = 100*ncomplete/ntasks
            if verbose:
                logger.info('percent complete = %2d   %d / %d tasks' % (percent,ntasks-taskrunning,ntasks))
        else:
            # It claims to not be running, but let's check anyway
            tstatus = taskstatus(label,key)
            ncomplete = len(tstatus)
            percent = 100*ncomplete/ntasks
            if verbose:
                logger.info('NOT Running  percent complete = %2d   %d / %d tasks' % (percent,ncomplete,ntasks))                
                
        # Are we done
        if noderunning==0 and taskrunning==0:
            done = True

def submit(tasks,label,nodes=5,cpus=64,alloc='sdss-np',qos='sdss-fast',shared=True,walltime='336:00:00',
           notification=False,memory=7500,numpy_num_threads=2,verbose=True,logger=None):
    """
    Submit a bunch of jobs

    tasks : table
      Table with the information on the tasks.  Must have columns of:
        cmd, outfile, errfile, dir (optional)

    """

    if logger is None:
        logger = dln.basiclogger()
    
    ppn = 64
    slurmpars = {'nodes':nodes, 'alloc':alloc, 'ppn':ppn, 'qos':qos, 'shared':shared,
                 'cpus':cpus, 'walltime':walltime, 'notification':False}

    username = getpwuid(getuid())[0]
    slurmdir = SLURMDIR+username+'/slurm/'
    if os.path.exists(slurmdir)==False:
        os.makedirs(slurmdir)

    # Generate unique key
    key = genkey()
    if verbose:
        logger.info('key = '+key)
    # make sure it doesn't exist yet

    # job directory
    jobdir = slurmdir+label+'/'+key+'/'
    if os.path.exists(jobdir)==False:
        os.makedirs(jobdir)

    if alloc=='sdss-np' and shared:
        partition = 'sdss-shared-np'
    else:
        partition = alloc

        
    # Start .slurm files

    # nodeXX.slurm that sources the procXX.slurm files    
    # nodeXX_procXX.slurm files with the actual commands in them

    # Figure out number of tasks
    ntasks = len(tasks)
    if ntasks < nodes*ppn:
        nodes = int(np.ceil(ntasks / 64))
        ncycle = 1
    else:
        ncycle = int(np.ceil(ntasks / (nodes * ppn)))

    # Add column to tasks table
    if isinstance(tasks,Table)==False:
        tasks = Table(tasks)
    tasks['task'] = -1
    tasks['node'] = -1
    tasks['proc'] = -1
        
    # Node loop
    tasknum = 0
    inventory = []
    for i in range(nodes):
        node = i+1
        nodefile = 'node%02d.slurm' % node
        nodename = 'node%02d' % node
        
        # Number of proc files
        nproc = ppn
        ntaskleft = ntasks-tasknum
        if ntaskleft < ppn:
            nproc = ntaskleft

        # Create the lines
        lines = []
        lines += ['#!/bin/bash']
        lines += ['# Auto-generated '+datetime.now().ctime()+' -- '+label+' ['+nodefile+']']
        lines += ['#SBATCH --account='+alloc]
        lines += ['#SBATCH --partition='+partition]
        lines += ['#SBATCH --nodes=1']
        lines += ['#SBATCH --ntasks=64']
        lines += [' ']
        lines += ['#SBATCH --mem-per-cpu='+str(memory)]
        lines += [' ']
        lines += ['#SBATCH --time='+walltime]
        lines += ['#SBATCH --job-name='+label]
        lines += ['#SBATCH --output='+label+'_%j.out']
        lines += ['#SBATCH --err='+label+'_%j.err']
        lines += ['# ------------------------------------------------------------------------------']
        lines += ['export OMP_NUM_THREADS=2']
        lines += ['export OPENBLAS_NUM_THREADS=2']
        lines += ['export MKL_NUM_THREADS=2']
        lines += ['export VECLIB_MAXIMUM_THREADS=2']
        lines += ['export NUMEXPR_NUM_THREADS=2']
        lines += ['# ------------------------------------------------------------------------------']
        lines += ['export CLUSTER=1']
        lines += [' ']
        for j in range(nproc):
            proc = j+1
            procfile = 'node%02d_proc%02d.slurm' % (node,proc)
            lines += ['source '+jobdir+procfile+' &']
        lines += ['wait']
        lines += ['echo "Done"']
        if verbose:
            logger.info('Writing '+jobdir+nodefile)
        dln.writelines(jobdir+nodefile,lines)
        
        # Create the proc files
        for j in range(nproc):
            if tasknum>=ntasks: break
            proc = j+1
            procname = 'proc%02d' % proc
            procfile = 'node%02d_proc%02d.slurm' % (node,proc)
            lines = []
            lines += ['# Auto-generated '+datetime.now().ctime()+' -- '+label+' ['+procfile+']']
            lines += ['cd '+jobdir]            
            # Loop over the tasks
            for k in range(ncycle):
                if tasknum>=ntasks: break
                lines += ['# ------------------------------------------------------------------------------']
                lines += ['echo "Running task '+str(tasknum+1)+' '+nodename+' '+procname+'" `date`']                
                if 'dir' in tasks.colnames:
                    lines += ['cd '+tasks['dir'][tasknum]]
                cmd = tasks['cmd'][tasknum]+' > '+tasks['outfile'][tasknum]+' 2> '+tasks['errfile'][tasknum]
                lines += [cmd]
                lines += ['echo "Task '+str(tasknum+1)+' '+nodename+' '+procname+' Completed" `date`']
                lines += ['echo "Done"']
                if os.path.exists(os.path.dirname(tasks['outfile'][tasknum]))==False:  # make sure output directory exists
                    try:
                        os.makedirs(os.path.dirname(tasks['outfile'][tasknum]))
                    except:
                        logger.info('Problems making directory '+os.path.dirname(tasks['outfile'][tasknum]))
                inventory += [str(tasknum+1)+' '+str(node)+' '+str(proc)]
                tasks['task'][tasknum] = tasknum+1
                tasks['node'][tasknum] = node
                tasks['proc'][tasknum] = proc
                tasknum += 1
            lines += ['cd '+jobdir]                            
            if verbose:
                logger.info('Writing '+jobdir+procfile)
            dln.writelines(jobdir+procfile,lines)

    # Create the "master" slurm file
    masterfile = label+'.slurm'
    lines = []
    lines += ['#!/bin/bash']
    lines += ['# Auto-generated '+datetime.now().ctime()+' ['+masterfile+']']
    lines += ['#SBATCH --account='+alloc]
    lines += ['#SBATCH --partition='+partition]
    lines += ['#SBATCH --nodes=1']
    lines += ['#SBATCH --ntasks=64']
    lines += [' ']
    lines += [' ']
    lines += ['#SBATCH --mem-per-cpu='+str(memory)]
    lines += [' ']
    lines += ['#SBATCH --time='+walltime]
    lines += ['#SBATCH --array=1-'+str(nodes)]
    lines += ['#SBATCH --job-name='+label]
    lines += ['#SBATCH --output='+label+'_%A[%a].out']
    lines += ['#SBATCH --err='+label+'_%A[%a].err']
    lines += ['# ------------------------------------------------------------------------------']
    lines += ['export OMP_NUM_THREADS=2']
    lines += ['export OPENBLAS_NUM_THREADS=2']
    lines += ['export MKL_NUM_THREADS=2']
    lines += ['export VECLIB_MAXIMUM_THREADS=2']
    lines += ['export NUMEXPR_NUM_THREADS=2']
    lines += ['# ------------------------------------------------------------------------------']
    lines += ['SBATCH_NODE=$( printf "%02d']
    lines += ['" "$SLURM_ARRAY_TASK_ID" )']
    lines += ['source '+jobdir+'node${SBATCH_NODE}.slurm']
    if verbose:
        logger.info('Writing '+jobdir+masterfile)
    dln.writelines(jobdir+masterfile,lines)

    # Write the number of tasks
    dln.writelines(jobdir+label+'.ntasks',ntasks)

    # Write the inventory file
    dln.writelines(jobdir+label+'_inventory.txt',inventory)

    # Write the tasks list
    tasks.write(jobdir+label+'_tasks.fits',overwrite=True)
    # Write the list of logfiles
    dln.writelines(jobdir+label+'_logs.txt',list(tasks['outfile']))
    
    # Now submit the job
    if verbose:
        logger.info('Submitting '+jobdir+masterfile)
    # Change to the job directory, because that's where the outputs will go
    curdir = os.path.abspath(os.curdir)
    os.chdir(jobdir)
    # Sometimes sbatch can fail for some reason
    #  if that happens, retry
    scount = 0
    success = False
    while (success==False) and (scount < 5):
        if scount>0:
            logger.info('Trying to submit to SLURM again')
        try:
            res = subprocess.check_output(['sbatch',jobdir+masterfile])
            success = True
        except:
            logger.info('Submitting job to SLURM failed with sbatch.')
            success = False
            tb = traceback.format_exc()
            logger.info(tb)
            time.sleep(10)
        scount += 1
    os.chdir(curdir)   # move back
    if type(res)==bytes: res = res.decode()
    res = res.strip()  # remove \n
    if verbose:
        logger.info(res)
    # Get jobid
    #  Submitted batch job 5937773 on cluster notchpeak
    jobid = res.split()[3]

    if verbose:
        logger.info('key = '+key)
        logger.info('job directory = '+jobdir)
        logger.info('jobid = '+jobid)
        
    return key,jobid


class Task(object):

    def __init__(self,cmd,outfile,errfile,directory=None):
        self.cmd = cmd
        self.outfile = outfile
        self.errfile = errfile
        self.directory = directory
        self.id = None
        self.done = None
        
class Queue(object):

    def __init__(self,label,nodes=5,alloc='sdss-np',shared=True,walltime='336:00:00',
                 notification=False,memory=7500,numpy_num_threads=2,verbose=True,logger=None):
        self.label = label
        self.nodes = nodes        
        self.tasks = []
        self.alloc = alloc
        self.shared = shared
        self.walltime = walltime
        self.notification = notification
        self.memory = memory
        self.numpy_num_thread = numpy_num_thread
        self.verbose = verbose
        if logger is None:
            logger = dln.basiclogger()
        self.logger = logger
        username = getpwuid(getuid())[0]
        self.user = username
        self.slurmdir = SLURMDIR+username+'/slurm/'
        self.jobid = None
        self.jobdir = None        
        self.submitted = False
        # Generated key
        self.key = genkey()

    def __repr__(self):
        self.logger.info('label = %d' % self.label)
        self.logger.info('user = %d' % self.user)
        self.logger.info('key = %s' % self.key)        
        self.logger.info('nodes = %d' % self.nodes)
        self.logger.info('alloc = %s' % self.alloc)
        self.logger.info('shared = %s' % self.shared)
        self.logger.info('walltime = %s' % self.walltime)
        self.logger.info('notification = %s' % self.notification)
        self.logger.info('memory = %s' % str(self.memory))
        self.logger.info('numpy_num_thread = %s' % str(self.numpy_num_thread))
        self.logger.info('verbose = %s' % self.verbose)
        self.logger.info('Ntasks = %d' % self.ntasks)
        self.logger.info('submitted = %s' % self.submitted)
        if self.submitted:
            self.logger.info('jobid = %d' % self.jobid)
            self.logger.info('jobdir = %s' % self.jobdir)
        
    @property
    def ntasks(self):
        return len(self.tasks)

    def append(self,cmd,outfile,errfile,directory=None):
        t = Task(cmd,outfile,errfile,directory=directory)
        t.id = self.ntasks+1
        self.tasks.append(t)

    def submit(self):
        pass

    def taskstatus(self):
        if self.submitted==False:
            print('Not submitted yet')
            return None
        tstatus = taskstatus(self.label,self.key)
        ntask = len(tstatus)
        ndone = np.sum(tstatus['done'])
        percent = 100*ndone/ntask
        return ndone, percent
        
    def slurmstatus(self):
        if self.submitted==False:
            print('Not submitted yet')
            return None
        state = slurmstatus(self.label,self.jobid)
        node = len(state)
        ndone = np.sum(state['done'])
        noderunning = node-ndone
        percent = 100*noderunning/node
        return ndone, percent
        
    def status(self):
        if self.submitted==False:
            print('Not submitted yet')
            return None
        return status(self.label,self.key,self.jobid)

    def kill(self):
        """ Kill the slurm jobs"""
        pass    
