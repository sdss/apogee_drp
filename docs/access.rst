***********
Data Access
***********

Here are the various ways that APOGEE data can be accessed.

Summary Files
-------------

Catalogs of summary information are available here:

`allStar-daily-apo25m.fits <https://data.sdss5.org/sas/sdsswork/mwm/apogee/spectro/redux/daily/summary/allStar-daily-apo25m.fits>`_

`allVisit-daily-apo25m.fits <https://data.sdss5.org/sas/sdsswork/mwm/apogee/spectro/redux/daily/summary/allVisit-daily-apo25m.fits>`_

SAS Access
----------

All of the data files can be accessed directly from the Utah server by logging in or by using the SAS webpage online:

`daily directory <https://data.sdss5.org/sas/sdsswork/mwm/apogee/spectro/redux/daily/>`_

Note, you need the sdss username and password to access the data online.

Database
--------

A lot of information is stored in the APOGEE data database ``apogeedb``.  This can be queried by users that are logged into the
Utah servers.  We are working on tools for offsite access.

The 13 tables are:
- plan: Information for each plan file.
- exposure: Information on all exposures.
- visit: Summary information for each visit spectrum.
- visit_latest: The latest information for each visit spectrum.
- star: Summary information for each unique star.
- star_latest: The latest information for each star.
- rv_visit: RV summary information for each visit spectrum.
- version: 

These 5 tables gives the status of the different processing steps:
- exposure_status: Status of exposure processing.
- daily_status: Status of the daily reduction.
- calib_status: Status of the calibration processing.
- visit_status: Status of visit processes at the plan level.
- rv_status: Status of RV processing.
  
The layout of the tables are availabl here:
`apogee_drp <https://github.com/sdss/sdssdb/tree/apogee_drp/schema/sdss5db/apogee_drp>`_


