*********
Log Files
*********

The APOGEE_DRP pipeline has extensive logging of every process and product.  Since there are many log files and many types of them, they are distributed throughout the APOGEE reduction directory structure.

The name of the main APOGEE reduction directory is ``$APOGEE_REDUX`` and the name of the reduction version is ``APRED``.  Therefore, all the directories discussed below will be relative to ``$APOGEE_REDUX/APRED/``.


Cron Logs
=========

Daily APOGEE reduction processing is started from an automatic cron job that runs every 30 minutes.  There is a night counter called currentmjd that is located in ``$APOGEE_REDUX/APRED/log/OBSERVATORY/currentmjd``, where ``OBSERVATORY`` is "apo" or "lco".  This one-line file contains the MJD date to which daily processing has been completed.  The cron job checks if there are data available for the next night and if any other daily processing jobs are running.  To avoid overloading the machines, only one daily processing job (started from cron) per telescope can run at a time.

The log file for each cron job is located: ``$APOGEE_REDUX/APRED/OBSERVATORY/cron/`` with names ``YYYY-MM-DD-HH:MM:SS-runapogee_cron.log``.  For example, ``2022-03-11-12:00:09-runapogee_cron.log``.

Most log files will either day "No data for MJD5=60096" or "One process of runapogeeapo already running.  Will try again when it is finished."

Here are some example cron log files:

`2023-06-07-10:30:01-runapogee_cron.log <_static/2023-06-07-10:30:01-runapogee_cron.log>`_

`2023-06-07-19:30:01-runapogee_cron.log <_static/2023-06-07-19:30:01-runapogee_cron.log>`_


Daily Processing Logs
=====================

The daily processing with ``runapogee`` produces a log file that is located in ``$APOGEE_REDUX/APRED/log/OBSERVATORY/`` for example ``redux/daily/log/apo/``.  The log files have names ``MJD.YYYYMMDDHHMMSS.log`` where ``YYYYMMDDHHMMSS`` is the timestamp.

Every job/process that gets run on the computer cluster gets its own log file.  This will all be listed in the daily log file before jobs are started on the cluster using Slurm.

Here are some example daily processing ``runapogee`` log files:

`59989.20230219231208.log <_static/59989.20230219231208.log>`_

`60049.20230426212215.log <_static/60049.20230426212215.log>`_

DRP Processing Logs
===================

DRP processing is when many nights of data are processed all at once.  This can happen for internal data releases (IPLs) or full data releases.  These files look similar to the daily processing ones and are also located in ``$APOGEE_REDUX/APRED/log/OBSERVATORY/`` for example ``redux/daily/log/apo/``.  Their names are ``apogeedrp.MJDBEG-MJDEND.YYYYMMDDHHMMSS.log``.

Here are some example ``apogeedrp`` log files:

`apogeedrp-59560-59619.20220209155504.log <_static/apogeedrp-59560-59619.20220209155504.log>`_

`apogeedrp-59608-59624.20220216104306.log <_static/apogeedrp-59608-59624.20220216104306.log>`_


Emails
======

At the end of both daily and DRP processing, an email is sent to apogee-pipeline-log@sdss.org.  This gives some basic summary information.

Here's an example of daily log::
  
  Subject: [apogee-pipeline-log 648] Daily APOGEE Reduction apo 59760

  Daily APOGEE Reduction apo 59760
  QA Webpage (MJD List)
  17/17 calibrations successfully processed
  76/82 exposures successfully processed
  6/8 visits successfully processed
  /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/daily/cal/apogee-n/plans/59760/apCalPlan-apogee-n-59760.yaml
  /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/daily/cal/apogee-n/plans/59760/apDarkPlan-apogee-n-59760.yaml
  /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/daily/visit/apo25m/103198/6037/59760/apPlan-6037-59760.yaml
  /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/daily/visit/apo25m/112359/6038/59760/apPlan-6038-59760.yaml
  /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/daily/visit/apo25m/112359/6039/59760/apPlan-6039-59760.yaml
  /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/daily/visit/apo25m/112359/6040/59760/apPlan-6040-59760.yaml
  /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/daily/visit/apo25m/112359/6041/59760/apPlan-6041-59760.yaml
  /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/daily/visit/apo25m/23104/6069/59760/apPlan-6069-59760.yaml
  233/251 RV+visit combination successfully processed``

The daily log is also attached to the email.

If for some reason the daily ``runapogee`` processing started by cron crashes, then an email message is sent that it crashed along with the exception::

  APOGEE daily reduction apo 60051 CRASHED!! timeout=timeout, check=True, File "/uufs/chpc.utah.edu/common/home/sdss50/software/pkg/miniconda/3.8.5_mwm/lib/python3.8/subprocess.py", line 512, in run raise CalledProcessError(retcode, process.args, subprocess.CalledProcessError: Command '['sbatch', '/scratch/general/nfs1/u0914350/pbs/apred/d74050d8-dc9d-11ed-a9fb-2cea7ff4461c/node01.slurm']' returned non-zero exit status 1.

A log file with the exception is attached.  Here's an example: `60051.log.crash <_static/60051.log.crash>`_


Calibration Logs
================

All of the master and daily calibration files live in the directory ``$APOGEE_REDUX/APRED/cal/INSTRUMENT/CALTYPE/`` where ``INSTRUMENT`` is "apogee-n" or "apogee-s" and ``CALTYPE`` is the calibration file type.  The various calibration types are: ``bpm``, ``darkcorr``, ``detector``, ``flatcorr``, ``flux``, ``fpi``, ``littrow``, ``lsf``, ``persist``, ``psf``, ``telluric``, ``trace``, and ``wave``.

The log files are in ``$APOGEE_REDUX/APRED/cal/INSTRUMENT/plans/MJD/`` for example ``redux/daily/cal/apogee-n/plans/59936``.  Here are some example log files:

`apFlux-43740001_pbs.20221223131431.log <_static/apFlux-43740001_pbs.20221223131431.log>`_

`apFPI-43740011_pbs.20221223131431.log <_static/apFPI-43740011_pbs.20221223131431.log>`_

`apPSF-43740005_pbs.20221223131431.log <_static/apPSF-43740005_pbs.20221223131431.log>`_

`apTelluric-59936-40030031_pbs.20221223131431.log <_static/apTelluric-59936-40030031_pbs.20221223131431.log>`_

`apWave-43740006_pbs.20221223131431.log <_static/apWave-43740006_pbs.20221223131431.log>`_

`apDailyWave-59936_pbs.20221223131431.log <_static/apDailyWave-59936_pbs.20221223131431.log>`_

If you know there is an issue with a calibration file for a given night, it's straightforward to go to the ``plans/MJD/`` directory to find the relevant log file.


AP3D Logs
=========

The products from the AP3D (collapsing the 3D cube to a 2D image) step are saved in the ``$APOGEE_REDUX/APRED/exposures/INSTRUMENT/MJD/`` directory (i.e. ``redux/daily/exposures/apogee-n/59796/``).  The log files live in the ``logs/`` subdirectory.  Here are some examples:

`ap3D-42340001_pbs.20220805120009.log <_static/ap3D-42340001_pbs.20220805120009.log>`_

`as3D-43450003_pbs.20221124131409.log <_static/as3D-43450003_pbs.20221124131409.log>`_

Again, if you know there was an issue at the AP3D level procelling, then it's straightward to go the exposurs logs directory to find the relevant log file.


Visit Logs
==========

Much of the processing of the science exposures is done in ``visit`` directories.  This does the AP2D level processing (spectral extraction) and AP1DVISIT processing (wavelength calibration, dither combination, etc.).  The products and log files live in ``$APOGEE_REDUX/APRED/visit/TELESCOPE/FIELD/CONFIGID/MJD/`` where TELESCOPE is "apo25m" or "lco25m". For example, ``redux/daily/visit/apo25m/103490/7980/59935/``.  The log files have names like ``apPLan-CONFIGID-MJD_pbs.YYYYMMDDHHMMSS.log``.

Here's an example:

`apPlan-7980-59935_pbs.20221222190511.log <_static/apPlan-7980-59935_pbs.20221222190511.log>`_

During the QA process, visit-level QA files are produced.  They are in the same directory as the visit log files.  Here's an example:

`apqa-7980-59935_pbs.20230118092842.log <_static/apqa-7980-59935_pbs.20230118092842.log>`_


RV Logs
=======

Durig the ``RV`` step the visit spectra get combined and ``Doppler`` is run to determine the radial velocities.  The products of the ``RV`` step live in ``$APOGEE_REDUX/APRED/stars/TELESCOPE/HEALPIXGRP/HEALPIX/``  Where ``HEALPIX`` is the nside=128 HEALPix number of the star's coordinates and ``HEALPIXGRP`` is ``HEALPIX`` divided by 1000.  For example, ``redux/daily/stars/apo25m/15/15634/``

Here are some example log files:

`apStar-daily-apo25m-2M21593609+5716051-59829_pbs.20220907184232.log <_static/apStar-daily-apo25m-2M21593609+5716051-59829_pbs.20220907184232.log>`_

`apStar-daily-apo25m-2M21593609+5716051-59829_pbs.20220907184232.log <_static/apStar-daily-apo25m-2M21593609+5716051-59829_pbs.20220907184232.log>`_

