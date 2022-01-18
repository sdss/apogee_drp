*****************
Quality Assurance
*****************

At the end of the APOGEE data reduction, a number of quality assurance (QA) plots and webpages are produced.  These
can be used to ascertain the quality of the data and the reduction.

Note, you need the sdss username and password to access the online QA pages.

MJD and Fields Lists
--------

One good starting points is the
`MJD view <https://data.sdss5.org/sas/sdsswork/mwm/apogee/spectro/redux/daily/qa/mjd.html>`_
page. Each row of the table pertains to a particular night of observing, providing links to the
on-site observers' logs, a list of the APOGEE exposures, the raw data directory, a nightly
QA page giving information about the observations and data reduction, a QA page for each 
observation, and links to allVisit and allStar summary files specific to the night in question. 
Links to instrument monitoring plots and full-survey allVisit and allStar files are provided 
at the top of the page. 

Another good starting point is the `Fields view <https://data.sdss5.org/sas/sdsswork/mwm/apogee/spectro/redux/daily/qa/fields.html>`_ page.
Each row of table pertains to a visit to a particular field center. The columns of the table
provide information about the field, the observation of it, and the achieved S/N. Plots of all 
of the field centers observed to date are provided at the top of the page along with links to 
instrument monitoring plots and full-survey allVisit and allStar files.

Visit Level
-----------

Each visit has two associated pages. The first is an "apQA" page that summarizes the individual exposures and combined visit, 
with various QA plots included.  Here is an example: 
`apQA-1279-59584.html <https://data.sdss5.org/sas/sdsswork/mwm/apogee/spectro/redux/daily/visit/apo25m/20882/1279/59584/html/apQA-1279-59584.html>`_. 
The second is an "apPlate" page that shows plots of the spectra from each visit, including skies, telluric standards, and science targets. Here is an example: 
`apPlate-1279-59584.html <https://data.sdss5.org/sas/sdsswork/mwm/apogee/spectro/redux/daily/visit/apo25m/20882/1279/59584/html/apPlate-1279-59584.html>`_.
Links to these pages are provided in the MJD and Fields summary pages described above.

Star Level
----------

Each observed star has its own star-level QA page that gives useful summary information in a table at the top, a plot
of the combined spectrum and best fitting Doppler Cannon Model, and a table giving information about an a plot of each
individual observation.  Here's an example page:
`2M06482624+0357058.html <https://data.sdss5.org/sas/sdsswork/mwm/apogee/spectro/redux/daily/stars/apo25m/91/91537/html/2M06482624+0357058.html>`_.
Links to these pages are provided in the "apPlate" QA pages described in the Visit Level section of this page.

Instrument Level
----------

Each of the APOGEE instruments has its own QA monitoring page showing plots of the long-term trends in things like quartz lamp
and dome flat brightness, wavelength calibration line positions, sky brightness, etc. The page for the APOGEE-N instrument is
here `apogee-n-monitor.html <https://data.sdss5.org/sas/sdsswork/mwm/apogee/spectro/redux/daily/monitor/apogee-n-monitor.html>`_,
the page for the APOGEE-S instrument is here 
`apogee-s-monitor.html <https://data.sdss5.org/sas/sdsswork/mwm/apogee/spectro/redux/daily/monitor/apogee-s-monitor.html>`_.



