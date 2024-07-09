import os
import time
import numpy as np
from astropy.io import fits
from astropy.table import Table
from ...utils import apload,plugmap,platedata,info,plan

chiptag = ['a','b','c']
BADERR = 1.0000000e+10

# dithershift()
# outcframe()
# dithercombine()
# fluxing
# visitoutput()
# sky.skysub()
# sky.telluric()



def calibrate_exposure(framenum,plantab,plugmap,logger=None,
                       clobber=False,verbose=False,nowrite=False):
    """
    This corrects a single APOGEE exposure/frame.
    It measures the dither shift, subtracts the sky and
    corrects for the telluric absorption.
    Finally, it writes the apCframe files.

    Parameters
    ----------
    framenum : int
       Expsore number.
    plantab : table
       Plan table with information about this visit.
    plugmap : table
       Plugmap information on the observed objects.
    logger : logging object
       Logging object.
    clobber : bool, optional
       Overwrite any existing files.  Default is False.
    verbose : bool, optional
       Verbose output to the screen.  Default is False.
    nowrite : bool, optional
       Don't outpout any apCframe files to disk.  Default is False.

    Returns
    -------
    frame_telluric : 
       Calibrated frame
    Unless nowrite=True is set, apCframe files are written to disk.

    Examples
    --------

    frame = calibrate_exposure(framenum,plantab,plugmap,logger)

    """

    t1 = time.time()

    load = apload.ApLoad(apred=plantab['redux'],telescope=plantab['telescope'])
    
    #------------------------------------------
    # Correcting and Calibrating the ap1D files
    #------------------------------------------
    cfiles = load.filename('Cframe',chip=chiptag,num=framenum,plate=plantab['plateid'],
                           mjd=plantab['mjd'],field=plantab['field'])
    ctest = np.array([os.path.exists(f) for f in cfiles])
    if clobber==False and np.sum(ctest)==3:
        print('Already done')
        return None

    logger.info(' 1d processing '+os.path.basename(files[0])+'{:8.2f}'.format(time.time()-t1))
            
    # Load the 1D files
    #--------------------
    frame0 = load.frame(cfiles)
    #APLOADFRAME,files,frame0,/exthead      # loading frame 1
    apaddpar,frame0,'LONGSTRN','OGIP 1.0'  # allows us to use long/continued strings
    
    # Fix INF and NAN
    for k in range(3):
        bdnan, = np.where(~np.isfinite(frame0[k]['flux']) |
                          ~np.isfinite(frame0[k]['err']))
        if len(bdnan)>0:
            frame0[k]['flux'][bdnan] = 0.0
            frame0[k]['err'][bdnan] = BADERR
            frame0[k]['mask'][bdnan] = 1   # bad
        # Fix ERR=0
        bdzero, = np.where(frame0[k]['err'] < 0)
        if len(bdzero)>0:
            frame0[k]['flux'][bdzero] = 0.0
            frame0[k]['err'][bdzero] = BADERR
            frame0[k]['mask'][bdzero] = 1

    # Add Wavelength and LSF information to the frame structure
    #---------------------------------------------------------
    # Loop through the chips
    for k in range(3):
        chtab = frame0[k]
        # Get the LSF calibration data
        lsfcoef,lhead = fits.getdata(lsffiles[k],header=True)
        if newwave is not None:
            del chtab['wcoef']
            del chtab['wavelength']
        # Add to the chip structure
        # Wavelength calibration data already added by ap2dproc with ap1dwavecal
        #if tag_exist(frame0.(0),'WCOEF') and not keyword_set(newwave) then begin
        if 'wcoef' in chtab.colnames: and newwave is None:
            if verbose:
                print('using WCOEF from 1D...')
            chtab.update({'lsffile':lsffiles[k],'lsfcoef':lsfcoef,'wave_dir':plate_dir,
                          'wavefile':wavefiles[k]})
        # Need wavelength information
        else:
            wcoef,whead = fits.getdata(wavefiles[k],header=True)
            chtab.update({'lsffile':lsffiles[k],'lsfcoef':lsfcoef,'wave_dir':plate_dir,
                          'wavefile':wavefiles[k],'wcoef':wcoef})
            # Now add this to the final FRAME structure
            frame['chip'+chiptag[k]] = chtab
    
    del frame0   # free up memory

    #----------------------------------
    # STEP 1:  Measure dither Shift
    #----------------------------------
    if verbose:
        print('STEP 1: Measuring the DITHER SHIFT with APDITHERSHIFT')
    # Not first frame, measure shift relative to 1st frame
    dither_commanded = frame[0]['header']['DITHPIX']
    if verbose:
        print('dither_commanded: ',dither_commanded)
    if j == 0 and verbose:
        print('ref_dither_commanded: ',ref_dither_commanded)
    print('nodither: ', nodither)
    if j > 0:
        if dither_commanded != 0 and np.abs(dither_commanded-ref_dither_commanded) > 0.002:
            nodither = False
    # Measure dither shift
    if (j > 0) and not nodither:
        ashift = [0.0,0.0]
        ashifterr = 0.0
        #APDITHERSHIFT,ref_frame,frame,ashift,ashifterr
        if 'platetype' in plantab.colnames:
            if plantab['platetype'] == 'sky' or plantab['platetype'] == 'cal':
                plot = True
                pfile = os.path.join(plate_dir,'plots','dithershift-'+framenum)
            else:
                #pfile=0 & plot=0
                plot = True
                pfile = os.path.join(plate_dir,'plots','dithershift-'+framenum)
        if plantab['platetype']=='single':
            nofit = True
        else:
            nofit = False
        shiftout = dithershift(ref_frame,frame,xcorr=True,pfile=pfile,plot=plot,
                               plugmap=plugmap,nofit=nofit,mjd=plantab['mjd'])
        shift = shiftout['shiftfit']
        shifterr = shiftout['shifterr']
        if verbose:
            print('Measured dither shift: ',ashift,shift)
    # First frame, reference frame
    else:
        # measure shift anyway
        if j > 0:
            shiftout = dithershift(ref_frame,frame,xcorr=True,pfile=pfile,plot=plot,
                                   plugmap=plugmap,nofit=nofit,mjd=plantab['mjd'])
            if verbose:
                print('Measured dither shift: ',shiftout['shiftfit'])
        # note reference frame wants to include sky and telluric!
        ref_frame = frame
        shift = [0.0,0.0]
        ashift = [0.0,0.0]
        shifterr = 0.0
        ashifterr = 0.0
        if dither_commanded != 0:
            ref_dither_commanded = dither_commanded
        print('Shift = 0.0')
        shiftout = {'type':'xcorr','shiftfit':np.zeros(2,float),'shfiterr':shifterr,
                    'chipshift':np.zeros((3,2),float),'chipfit':np.zeros(4,float)}
    apaddpar,frame,'APDITHERSHIFT: Measuring the dither shift',/history
    if shift[0] == 0.0: apaddpar,frame,'APDITHERSHIFT: This is the REFERENCE FRAME',/history
    apaddpar,frame,'DITHSH',shift[0],' Measured dither shift (pixels)'
    apaddpar,frame,'DITHSLOP',shift[1],' Measured dither shift slope (pixels/fiber)'
    apaddpar,frame,'EDITHSH',shifterr,' Dither shift error (pixels)'
    #apaddpar,frame,'ADITHSH',ashift,' Measured dither shift (pixels)'
    #apaddpar,frame,'AEDITHSH',ashifterr,' Dither shift error (pixels)'
    ADD_TAG,frame,'SHIFT',shiftout,frame_shift
        
    logger.info('  dithershift '+'{:8.2f}{:8.2f}'.format(time.time()-t1,time.time()-t0))
    if 'platetype'] in plantab.colnames:
        if plantab['platetype'] not in ['normal','single','twilight']:
            continue

    #----------------------------------
    # STEP 2:  Wavelength Calibrate
    #----------------------------------
    # THIS IS NOW DONE AS PART OF AP2DPROC, USING PYTHON ROUTINES
    if ap1dwavecal:
        if verbose:
            print('STEP 2: Wavelength Calibrating with AP1DWAVECAL')
        plotfile = plate_dir+'/plots/pixshift_chip-'+framenum
        if dithonly:
            ap1dwavecal_refit(frame,frame_wave,plugmap=plugmap,verbose=True,plot=True,pfile=plotfile)
        plotfile = plate_dir+'/plots/pixshift-'+framenum
        if plantab['platetype'] == 'twilight':
            ap1dwavecal(frame_shift,frame_wave,verbose=True,plot=True,pfile=plotfile)
        else:
            ap1dwavecal(frame_shift,frame_wave,plugmap=plugmap,verbose=True,plot=True,pfile=plotfile)
        del frame  # free up memory
        logger.info('  wavecal '+'{:8.2f}{:8.2f}'.format(time.time()-t1,time.time()-t0))
    else:
        frame_wave = frame_shift

    #----------------------------------
    # STEP 3:  Airglow Subtraction
    #----------------------------------
    if verbose:
        print('STEP 3: Airglow Subtraction with APSKYSUB')
    sky.skysub(frame_wave,plugmap,frame_skysub,subopt=1,error=skyerror,force=force)
    #if n_elements(skyerror) gt 0 and planstr.platetype ne 'twilight' then begin
    #  stop,'halt: APSKYSUB Error: ',skyerror
    #  apgundef,frame_wave,frame_skysub,skyerror
    #endif
    del frame_wave  # free up memory
    logger.info('  airglow '+'{:8.2f}{:8.2f}'.format(time.time()-t1,time.time()-t0))
    if 'platetype' in plantab.colnames:
        if plantab['platetype'] not in ['normal','single','twilight']:
            continue

    #----------------------------------
    # STEP 4:  Telluric Correction
    #----------------------------------
    if verbose:
        print('STEP 4: Telluric Correction with APTELLURIC')
    if plantab['platetype'] == 'single':
        starfit = 2
        single = 1
        pltelstarfit = 1
    elif plantab['platetype'] == 'twilight':
        starfit = 0
    else:
        starfit = 1
        single = 0
        pltelstarfit = 0
        visittab = None
    if 'pltelstarfit'] in plantab.colnames:
        pltelstarfit = plantab['pltelstarfit']
    if 'usetelstarfit'] in plantab.colnames:
        usetelstarfit = 1
    else:
        usetelstarfit = 0
    if 'maxtellstars' in plantab.colnames:
        maxtellstars = plantab['maxtellstars']
    else:
        maxtellstars = 0
    if 'tellzones' in plantab.colnames:
        tellzones = plantab['tellzones']
    else:
        tellzones = 0
    sky.telluric(frame_skysub,plugmap,frame_telluric,tellstar,starfit=starfit,
                 single=single,pltelstarfit=pltelstarfit,usetelstarfit=usetelstarfit,
                 maxtellstars=maxtellstars,tellzones=tellzones,specfitopt=1,
                 plots_dir=plots_dir,error=telerror,save=True,preconv=True,visittab=visittab,
                 test=test,force=force)
    tellstar['im'] = plantab['apexp'][j]['name']
    ADD_TAG,frame_telluric,'TELLSTAR',tellstar,frame_telluric
    if len(alltellstar)==0:
        alltellstar = tellstar
    else:
        alltellstar.append(tellstar)
    #if n_elements(telerror) gt 0 and planstr.platetype ne 'single' and not keyword_set(force) then begin
    #  print('not halted: APTELLURIC Error: ',telerror)
    #  ntellerror+=1
    #  apgundef,frame_skysub,frame_telluric,telerror
    #  goto, BOMB1
    del frame_skysub  # free up memory
    logger.info(logfile,'  telluric '+'{:8.2f}{:8.2f}'.format(time.time()-t1,time.time()-t0))

    #-----------------------
    # Output apCframe files
    #-----------------------
    if nowrite==False:
        if verbose:
            print('Writing output apCframe files')
        outfiles = load.filename('Cframe',chip=chiptag,num=framenum,
                                 plate=plantab['plateid'],mjd=plantab['mjd'],
                                 field=plantab['field'])
        outcframe(frame_telluric,plugmap,outfiles,verbose=False)

    return frame_telluric


def ap1dvisit(planfiles,clobber=False,verbose=False,newwave=None,
              test=None,mapper_data=None,halt=None,dithonly=None,
              ap1dwavecal=None,force=False):
    """
    This program processes 1D APOGEE spectra.  It does dither
    combination, wavelength calibration and sky correction.

    Parameters
    ----------
    planfiles : str or list
       Input list of plate plan files
    clobber : bool, optional
       Don't use the apCframe files previously created (if it exists)
    verbose : bool, optional
       Print a lot of information to the screen

    Returns
    -------
    1D dither combined, wavelength calibrated, and sky corrected
    spectra.  4Kx300x5 with planes:
         1. flux (sky subtracted, absorption/flux corrected)
         2. (wavelength, if needed)
         3. flux error
         4. sky estimate shifted to wavelength sampling of object
         5. telluric/flux response shifted to wavelength sampling of object
         6. flags
    The names are apSpec-[abc]-PLATE4-MJD5.fits

    Examples
    --------

    ap1dvisit(planfiles)

    Written by D.Nidever  Mar. 2010
    Modifications J. Holtzman 2011+
    Translated to Python by D. Nidever  2024
    """

    #common telluric,convolved_telluric

    if ap1dwavecal is not None:
        newwave = True

    t0 = time.time()
    nplanfiles = len(np.atleast_1d(planfiles))

    print('')
    print('RUNNING AP1DVISIT')
    print('')
    print(len(nplanfiles),' PLAN files')
    
    chiptag = ['a','b','c']

    #--------------------------------------------
    # Loop through the unique PLATE Observations
    #--------------------------------------------
    for i in range(nplanfiles):
        planfile = planfiles[i]
        print('')
        print('=========================================================================')
        print('{:d}/{:d}  Processing Plan file {:}'.format(i+1,nplanfiles,planfile))
        print('=========================================================================')

        # Load the plan file
        #--------------------
        print('')
        print('Plan file information:')
        plantab = aploadplan(planfile,verbose=False)
        if plantab['mjd'] >= 59556:
            fps = True
        else:
            fps = False
        if 'field' not in plantab.colnames:
            plantab['field'] = '     '
            plantab['field'] = apogee_field(0,plantab['plateid'])

	# Get APOGEE directories
        load = apload.ApLoad(apred=plantab['redux'],telescope=plantab['telescope'])
        #dirs = getdir(apogee_dir,cal_dir,spectro_dir,apred_vers=apred_vers,datadir=datadir)
        #logfile = apogee_filename('Diag',plate=plantab['plateid'],mjd=plantab['mjd'])
        logfile = load.filename('Diag',plate=plantab['plateid'],mjd=plantab['mjd'])

        # Only process "normal" plates
        if 'platetype' in plantab.colnames:
            normal_plate_types = ['normal','twilight','sky','single','cal']
            if plantabe['platetype'] not in normal_plate_types:
                continue

        if 'survey' in plantab.colnames:
            survey = plantab['survey']
        else:
            if plantab['plateid'] >= 15000:
                survey = 'mwm'
            else:
                survey = 'apogee'
            if plantab['plateid'] == 0:  # cals
                survey = 'mwm'

        # Load the Plug Plate Map file
        #------------------------------
        print('')
        print('Plug Map file information:')
        if 'force' in plantab.colnames and force is None:
            force = plantab['force']
        if 'fixfiberid'] in plantab.colnames:
            fixfiberid = plantab['fixfiberid']
        if type(fixfiberid) is str and np.atleast_1d(fixfiberid).size==1:
            if fixfiberid.strip()=='null' or fixfiberid.strip()=='none':
                fixfiberid = None
        if 'badfiberid' in plantab.colnames:
            badfiberid = plantab['badfiberid']
        if type(badfiberid) is str and np.atleast_1d(badfiberid).size==1:
            if badfiberid.strip()=='null' or badfiberid.strip()=='none':
                badfiberid = None

        # Check for existing plate data file
        if plantab['platetype']=='single':
            plugfile = getenv('APOGEEREDUCE_DIR')+'/data/plPlugMapA-0001.par'
            plugmap = platedata.getdata(plantab['plateid'],plantab['mjd'],
                                        obj1m=plantab['apexp'][0]['singlename'],
                                        starfiber=plantab['apexp'][0]['single'],
                                        fixfiberid=fixfiberid)
        elif plantab['platetype']=='twilight':
            plugmap = platedata.getdata(plantab['plateid'],plantab['mjd'],twilight=True)
        elif plantab['platetype']=='cal':
            print('no plugmap for cal frames')
        else:
            plugfile = plantab['plugmap']
            plugmap = platedata.getdata(plantab['plateid'],plantab['mjd'],
                                        plugid=plantab['plugmap'],
                                        fixfiberid=fixfiberid,badfiberid=badfiberid,
                                        mapper_data=mapper_data)

        #if n_elements(plugerror) gt 0 then goto,BOMB

        if plantab['platetype']=='cal':
            plugmap['mjd'] = plantab['mjd']   # enter MJD from the plan file

        # Get objects
        if plantab['platetype']=='single':
            obj, = np.where((plugmap['fiberdata']['objtype'] != 'SKY') &
                            (plugmap['fiberdata']['spectrographId']==2))
        else:
            obj, = np.where((plugmap['fiberdata']['objtype'] != 'SKY') &
                            (plugmap['fiberdata']['spectrographId']==2) &
                            (plugmap['fiberdata']['mag'][1] > 7.5))

        # Check if the calibration files exist
        #-------------------------------------
        makecal.makecal(lsf=plantab['lsfid'],full=True)
        wavefiles = load.filename('Wave',chip=chiptag,num=plantab['waveid'])
        # We are now using dailywave files with MJD names
        if plantab['waveid'] < 1e7:
            wavefiles = os.path.dirname(wavefiles[0])
            wavefiles += load.prefix+'Wave-'+chiptag+'-'+str(plantab['waveid'])+'.fits'
            wavetest = np.array([os.path.exists(f) for f in wavefiles])
            lsffiles = load.filename('LSF',chip=chiptag,num=plantab['lsfid'])
            lsftest = np.array([os.path.exists(f) for f in lsffiles])
            if np.sum(wavetest) == 0 or np.sum(lsftest) == 0:
                bd1, = np.where(wavetest==0)
                if len(bd1) > 0:
                    print(wavefiles[bd1],' NOT FOUND')
                bd2 = where(lsftest==0)
                if len(bd2) > 0:
                    print(lsffiles[bd2],' NOT FOUND')
                continue

        # Do the output directories exist?
        plate_dir = load.filename('Plate',mjd=plantab['mjd'],
                                  plate=plantab['plateid'],chip='a',
                                  field=plantab['field'],dir=True)
        if os.path.exists(plate_dir)==False:
            os.makedirs(plate_dir)
        sdir = plate_dir.split('/')
        if load.telescope != 'apo1m' and plantab['platetype'] != 'cal':
            if os.path.exists(spectro_dir+'/plates/'+sdir[-2]):
                srcfile = '../'+sdir[-5]+'/'+sdir[-4]+'/'+sdir[-3]+'/'+sdir[-2]
                destfile = spectro_dir+'/plates/'+sdir[-2]
                os.link(srcfile,destfile)
        #cloc=strtrim(string(format='(i)',plugmap.locationid),2)
        #file_mkdir,spectro_dir+'/location_id/'+dirs.telescope
        #if ~file_test(spectro_dir+'/location_id/'+dirs.telescope+'/'+cloc) then file_link,'../../'+s[-5]+'/'\
        #    +s[-4]+'/'+s[-3],spectro_dir+'/location_id/'+dirs.telescope+'/'+cloc


        # Are there enough files
        nframes = len(plantab['apexp'])
        #if nframes lt 2 then begin
        #  print,'Need 2 OBSERVATIONS/DITHERS to Process'
        #  goto,BOMB
        #endif

        # Start the plots directory
        plots_dir = plate_dir+'/plots/'
        if os.path.exists(plots_dir)==False:
            os.makedirs(plots_dir)
        
        visittab = []
        alltellstar = []
        allframes = []

        # Do we already have apPlate file?
        filename = load.filename('Plate',chip='c',mjd=plantab['mjd'],
                                 plate=plantab['plateid'],field=plantab['field'])
        if os.path.exists(filename) and clobber==False:
            print('File already exists: ', filename)
            #goto,dorv
        else:
            print('cannot find file: ', filename)

        # Process each frame
        #-------------------
        dt = [('index',int),('framenum',int),('shift',float),('shifterr',float),('shiftfit',float,2),
              ('chipshift',float,(3,2)),('chipfit',float,4),('pixshift',float),('sn',float)]
        shifttab = np.zeros(nframes,dtype=np.dtype(dt))
        shifttab['index'] = -1
        shifttab['shift'] = np.nan
        shifttab['shifterr'] = np.nan
        shifttab['sn'] = -1.0

        # Assume no dithering until we see that a dither has been commanded from the
        #   header cards
        nodither = 1
        ntellerror = 0

        # Loop over the exposures/frames
        #-------------------------------
        for j in range(frames):
            t1 = time.time()

            # for ASDAF plate, fix up the plugmap structure
            if plantab['platetype'] == 'asdaf':
                allind, = np.where(plugmap['fiberdata']['spectrographId']==2)
                plugmap['fiberdata'][allind]['objtype'] = 'SKY'
                star, = np.where((plugmap['fiberdata']['spectrographId']==2) &
                                 (plugmap['fiberdata']['fiberid']==plantab['apexp'][j]['single']))
                plugmap['fiberdata'][star]['objtype'] = 'STAR'
                plugmap['fiberdata'][star]['tmass_style'] = plantab['apexp'][j]['singlename']
                plugmap['fiberdata'][star]['mag'] = np.zeros(5,float)+99.
                if 'hmag' in plantab.colnames:
                    plugmap['fiberdata'][star]['mag'][1] = plantab['hmag']
                else:
                    plugmap['fiberdata'][star]['mag[']1] = 5

            # Make the filenames and check the files
            rawfiles = load.filename('R',chip=chiptag,num=plantab['apexp'][j]['name'])
            rawinfo = info.info(rawfiles,verbose=False)    # this returns useful info even if the files don't exist
            framenum = rawinfo[0]['fid8']   # the frame number
            files = load.filename('1D',chip=chiptag,num=framenum)
            einfo = info.info(files,verbose=False)
            okay = (einfo['exists'] & einfo['sp1dfmt'] & einfo['allchips'] &
                    (einfo['mjd5'] == plantab['mjd']) & ((einfo['naxis']==3) | (einfo['exten']==True)))
            if np.sum(okay) == 0:
                bd, = np.where(np.array(okay)==False)
                raise Exception('halt: There is a problem with files: '+','.join(files)[bd])

            print('')
            print('-----------------------------------------')
            print('{:d}/{:d}  Processing Frame Number >>{:}<<'.format(j+1,nframes,framenum))
            print('-----------------------------------------')

            dum = calibrate_exposure(framenum,plantab,plugmap,logger,
                                     clobber=clobber,verbose=verbose)

            #---------------------------------------------
            # Using the apCframe files previously created
            #---------------------------------------------
            
            # Make the filenames and check the files
            # Cframe files
            cfiles = load.filename('Cframe',chip=chiptag,num=framenum,plate=plantab['plateid'],
                                   mjd=plantab['mjd'],field=plantab['field'])
            cinfo = info.info(cfiles,verbose=False)
            okay = (cinfo['exists'] & cinfo['allchips'] & (cinfo['mjd5']==plantab['mjd')) &
                    ((cinfo['naxis']==3) | (cinfo['exten']==1)))
            if np.sum(okay) < 1:
                bd, = np.where(np.array(okay)==False)
                raise Exceptioon('halt: There is a problem with files: '+','.join(cfiles[bd]))

            print('Using apCframe files: '+cfiles)

            # Load the apCframe file
            frame_telluric = load.frame(cfiles)
            #APLOADCFRAME,cfiles,frame_telluric,/exthead

            # Get the dither shift information from the header
            if j == 0:
                ref_dither_commanded = frame_telluric[0]['header']['DITHPIX']
                if ref_frame is None:
                    ref_frame = frame_telluric
            else:
                dither_commanded = frame_telluric[0]['header']['DITHPIX']
                if dither_commanded != 0 and abs(dither_commanded-ref_dither_commanded) > 0.002:
                    nodither = False
            shift = frame_telluric[0]['header']['DITHSH']
            shifterr = frame_telluric[0]['header']['EDITHSH']
            pixshift = frame_telluric[0]['header']['MEDWSH']

            # Add to the ALLFRAMES structure
            #--------------------------------
            if len(allframes) == 0:
                allframes = frame_telluric
            else:
                allframes.append(frame_telluric)

            # Update SHIFTTAB
            shifttab[j]['index'] = j
            shifttab[j]['framenum'] = framenum
            shifttab[j]['shift'] = shift
            shifttab[j]['shifterr'] = shifterr
            shifttab[j]['pixshift'] = pixshift
            shifttab[j]['shiftfit'] = frame_telluric['shift']['shiftfit']
            shifttab[j]['chipshift'] = frame_telluric['shift']['chipshift']
            shifttab[j]['chipfit'] = frame_telluric['shift']['chipfit']
            # Get S/N of brightest non-saturated object, just for sorting by S/N
            if plantab['platetype']=='single':
                obj, = np.where((plugmap['fiberdata']['objtype'] != 'SKY') &
                                (plugmap['fiberdata']['spectrographId'] == 2))
            else:
                obj, = np.where((plugmap['fiberdata']['objtype'] != 'SKY') &
                                (plugmap['fiberdata']['spectrographid'] == 2) &
                                (plugmap['fiberdata']['mag'][1] > 7.5) &
                                (plugmap['fiberdata']['fiberid'] != 195))
            hmag = plugmap['fiberdata'][obj]['mag'][1]
            isort = np.argsort(hmag)
            ibright = obj[isort[0]]
            fbright = 300-plugmap['fiberdata'][ibright]['fiberid']
            shifttab[j]['sn'] = np.median(frame_telluric[1]['flux'][:,fbright]/frame_telluric[1].['err'][:,fbright])

            # Instead of using S/N of brightest object, which is subject to any issue with that object,
            #  use median frame zeropoint instead
            if plantab['platetype'] != 'single':
                fiberloc = 300-plugmap['fiberdata'][obj]['fiberid']
                zero = np.median([hmag+2.5*np.log10(np.median(frame_telluric[1]['flux'][:,fiberloc],axis=0))])
                shifttab[j]['sn'] = zero
            del frame_telluric  # free up memory

        if dithonly:
            return

    # Write summary telluric file
    if plantab['platetype'] == 'single':
        tellstarfile = plantab['plate_dir']+'/apTellstar-'+str(plantab['mjd'])+'-'+str(plantab['name'])+'.fits'
        fits.writeto(tellstatfile,alltellstar,overwrite=True)
    elif plantab['platetype'] == 'normal':
        tellstarfile = load.filename('Tellstar',plate=plantab['plateid'],
                                     mjd=plantab['mjd'],field=plantab['field'])
        fits.writeto(tellstarfile,alltellstar,overwrite=True)
    t1 = time.time()

    if 'platetype' in plantab.colnames:
        if plantab['platetype'] in ['normal','single','twilight']:
            continue

    # Remove frames that had problems from SHIFTTAB
    if nodither:
        minframes = 1
    else:
        minframes = 2
    if nodither == 0 or nframes > 1:
        bdframe, = np.where(shifttab['index'] == -1)
        if (nframes-len(bdframe)) < minframes:
            raise Exception('halt: Error: dont have two good frames to proceed')
        if len(bdframe)>0:
            shiftab = np.delete(bdframe,shifttab)

    # stop with telluric errors
    if ntellerror>0:
        print(ntellerror,' frames had APTELLURIC errors')
        raise Exception('halt: '+str(ntellerror)+' frames had APTELLURIC errors')

    #----------------------------------
    # STEP 5:  Dither Combining
    #----------------------------------
    print('STEP 5: Combining DITHER FRAMES with APDITHERCOMB')
    dithercombine(allframes,shifttab,pairtab,plugmap,combframe,median=True,
                  newerr=True,npad=50,nodither=nodither,avgwave=True,verbose=True)
    logger.info(' dithercomb '+os.path.dirname(planfile)+'{:8.2f}{:8.2f}'.format(time.time()-t1,time.time()-t0))
    if len(pairtab)==0 and nodither==0:
        raise Exception('halt: Error: no dither pairs')


    #----------------------------------
    # STEP 6:  Flux Calibration
    #----------------------------------
    print('STEP 6: Flux Calibration with AP1DFLUXING')
    if plantab['platetype'] != 'single':
        fiberloc = 300-plugmap['fiberdata'][obj]['fiberid']
        zero = np.median([hmag+2.5*np.log10(np.median(combframe[1]['flux'][:,fiberloc],axis=0))])
    else:
        zero = 0.
    finalframe = fluxing(combframe,plugmap)
    
    #----------------------------------------------------
    # Output apPlate frames and individual visit spectra, and load apVisit headers with individual star info
    #----------------------------------------------------
    print('Writing output apPlate and apVisit files')
    if plantab['platetype']=='single':
        single = True
    else:
        single = False
    mjdfrac = None
    if 'mjdfrac' in plantab.colnames and plantab['mjdfrac']==1:
        mjdfrac = finalframe[0]['header']['JD-MID']-2400000.5
    visitoutput(finalframe,plugmap,shifttab,pairtab,single=single,
                verbose=False,mjdfrac=mjdfrac,survey=survey)
    logger.info(' output '+os.pathbasename(planfile)+'{:8.2f}{:8.2f}'.format(time.time()-t1,time.time()-t0))

    #---------------
    # Radial velocity measurements for this visit
    #--------------
    #dorv:
    if 'platetype' in plantab.colnames:
        if plantab['platetype'] in ['normal','single']:
            continue
    print('Radial velocity measurements')
    locid = plugmap['locationid']
    visittabfile = load.filename('VisitSum',plate=plantab['plateid'],mjd=plantab['mjd'],
                                 reduction=plugmap['fiberdata'][obj]['tmass_style'],
                                 field=plantab['field'])
    if 'mjdfrac' in plantab.colnames and plantab['mjdfrac']==1:
        cmjd = str(plantab['mjd']).strip()
        s = visittabfile.split(cmjd)
        visittabfile = s[0] + '{:8.2f}'.format(mjdfrac) + s[1]
        #s = strsplit(visittabfile,cmjd,/extract,/regex)
        #visittabfile = s[0]+string(format='(f8.2)',mjdfrac)+s[1]
    if os.path.exists(visittabfile) and clobber==False:
        print('File already exists: ', visittabfile)
        return
    outdir = os.path.dirname(visittabfile)
    if os.path.exists(outdir)==False:
        os.makedirs(outdir)

    objtype = np.char.array(plugmap['fiberdata']['objtype']).strip()
    if fps:
        objind, = np.where((plugmap['fiberdata']['spectrographid'] == 2) &
                           (plugmap['fiberdata']['holetype'] == 'OBJECT') &
                           (plugmap['fiberdata']['assigned'] == 1) &
                           (plugmap['fiberdata']['on_target'] == 1) &
                           (plugmap['fiberdata']['valid'] == 1) &
                           (objtype == 'STAR' | objtype == 'HOT_STD') &
                           objtype != 'SKY' & objtype != 'none' &
                           np.char.array(plugmap['fiberdata']['tmass_style']).strip() !- '2MNone')
    else:
        objind, = np.where((plugmap['fiberdata']['spectrographid'] == 2) &
                           (plugmap['fiberdata']['holetype'] == 'OBJECT') &
                           (objtype == 'STAR' | objtype == 'HOT_STD') &
                           objtype != 'SKY' & objtype != 'none' &
                           np.char.array(plugmap['fiberdata']['tmass_style']).strip() !- '2MNone')
    objdata = plugmap['fiberdata'][objind]
    obj = plugmap['fiberdata'][objind]['tmass_style']

    if single:
        if 'mjdfrac' in plantab.colnames and plantab['mjdfrac']==1:
            mjd = finalframe[0]['header']['JD-MID']]-2400000.5
        else:
            mjd = plantab['mjd']
        visitfile = apread('Visit',plate=plantab['plateid'],mjd=mjd,fiber=objdata[0]['fiberid'],
                           reduction=obj,field=plantab['field'])
        header0 = visitfile[0]['hdr']
    else:
        finalframe = apread('Plate',mjd=plantab['mjd'],plate=plantab['plateid'],field=plantab['field'])
        header0 = finalframe[0]['hdr']

    if 'plateid' in plugmap.colnames:
        plate = plugmap['plateid']
    else:
        plate = plugmap['plate']
    mjd = plugmap['mjd']
    platemjd5 = str(plate)+'-'+str(mjd)

    # Loop over the objects
    allvisittab = []
    for istar in range(nobjind):
        visitfile = apogee_filename('Visit',plate=plantab['plateid'],mjd=plantab['mjd'],
                                    fiber=objdata[istar]['fiberid'],reduction=obj,field=plantab['field'])
        if 'mjdfrac' in plantab.colnames and plantab['mjdfrac']==1:
            cmjd = str(mjd)
            s = visitfile.split(cmjd)
            visitfile = s[0]+cmjd+s[1] + '{:8.2f}'.format(mjdfrac) + s[2]
            #s = strsplit(visitfile,cmjd,/extract,/regex)
            #visitfile = s[0]+cmjd+s[1]+string(format='(f8.2)',mjdfrac)+s[2]

        vtb = [('apogee_id',str,25),('target_id',str,50),('file',str,300),('uri',str,300),
               ('apred_vers',str,50),('fiberid',int),('plate',str,50),('exptime',float),('nframes',int),
               ('mjd',int),('telescope',str,10),('survey',str,20),('field',str,50),('design',str,50),
               ('programname',str,20),('objtype',str,20),('assigned',int),('on_target',int),('valid',int),
               ('ra',float),('dec',float),('glon',float),('glat',float),('healpix',int),('jmag',float),
               ('jerr',float),('hmag',float),('herr',float),('kmag',float),('kerr',float),('src_h',str,20),
               ('pmra',float),('pmdec',float),('pm_src',str,20),('apogee_target1',int),('apogee_target2',int),
               ('apogee_target3',int),('apogee_target4',int),('sdss5_target_pks',str,100),
               ('sdss5_target_catalogids',str,300),('sdss5_target_carton_pks',str,300),('sdss5_target_cartons',str,1000),
               ('sdss5_target_flagshex',str,1000),('brightneicount',int),('brightneiflag',int),('brightneiflux',float),
               ('catalogid',int),('sdss_id',int),('ra_sdss_id',float),('dec_sdss_id',float),('gaia_release',str,10),
               ('gaia_sourceid',int),('gaia_plx',float),('gaia_plx_error',float),('gaia_pmra',float),
               ('gaia_pmra_error',float),('gaia_pmdec',float),('gaia_pmdec_error',float),('gaia_gmag',float),
               ('gaia_gerr',float),('gaia_bpmag',float),('gaia_bperr',float),('gaia_rpmag',float),('gaia_rperr',float),
               ('sdssv_apogee_target',int),('firstcarton',str,50),('cadence',str,20),('program',str,50),
               ('category',str,50),('targflags',str,100),('snr',float),('starflag',int),('starflags',str,200),
               ('dateobs',str,50),('jd',float)]
        visittab = Table(np.zeros(1,dtype=np.dtype(vtb)))
        visittab['apogee_id'] = obj[istar]
        visittab['target_id'] = objdata[istar]['object']
        visittab['file'] = os.path.basename(visitfile)
        # URI is what you need to get the file, either on the web or at Utah
        mwm_root = os.environ['MWM_ROOT']
        visittab['uri'] = visitfile[len(mwm_root)+1:]
        visittab['apred_vers'] = apred_vers
        visittab['fiberid'] = objdata[istar]['fiberid']
        visittab['plate'] = str(plantab['plateid']).strip()
        visittab['field'] = str(plantab['field']).strip()
        if 'designid' in plantab.colnames:
            visittab['design'] = str(plantab['designid']).strip()
        else:
            visittab['design'] = -999
        visittab['exptime'] = finalframe[0]['hdr']['exptime']
        visittab['nframes'] = nframes
        visittab['mjd'] = plantab['mjd']
        visittab['telescope'] = load.telescope
        # Copy over all relevant columns from plugmap/plateHoles/catalogdb
        if 'gaia_sourceid' in objdata.colnames:
            if objdata[istar]['gaia_sourceid']=='':
                objdata[istar]['gaia_sourceid'] = '-1'
        for c in visittab.colnames:
            visittab[c] = objdata[istar][c]
        #STRUCT_ASSIGN,objdata[istar],visittab,/nozero
        #GLACTC,visittab.ra,visittab.dec,2000.0,glon,glat,1,/deg
        coo = SkyCoord(visittab['ra'],visittab['dec'],unit='deg',frame='icrs')
        visittab['glon'] = coo.galactic.l.deg
        visittab['glat'] = coo.galactic.b.deg
        visittab['apogee_target1'] = objdata[istar]['target1']
        visittab['apogee_target2'] = objdata[istar]['target2']
        visittab['apogee_target3'] = objdata[istar]['target3']
        visittab['apogee_target4'] = objdata[istar]['target4']

        # SDSS-V flags
        if plantab['plateid'] >= 15000:
            visittab['targflags'] = targflag(visittab['sdssv_apogee_target0'],survey=survey)
        # APOGEE-1/2 flags
        else:
            visittab['targflags'] = targflag(visittab['apogee_target1'],visittab['apogee_target2'],
                                             visittab['apogee_target3'],visittab['apogee_target4'],survey=survey)
        visittab['survey'] = survey
        visittab['field'] = str(plugmap['field']).strip()
        visittab['programname'] = plugmap['programname']

        # Get a few things from apVisit file (done in aprv also, but not
        #   if that is skipped....)
        #apgundef,str
        vtab = load.visit(visitfile)
        #APLOADVISIT,visitfile,vstr
        visittab['dateobs'] = vtab['dateobs']
        if 'JDMID' in vtab.colnames:
            visittab['jd'] = vtab['jdmid']
            aprvjd = vtab['jdmid']
        else:
            visittab['jd'] = vtab['jd']
            aprvjd = vtab['jd']            
        visittab['snr'] = vtab['snr']
        visittab['starflag'] = vtab['starflag']
        visittab['starflags'] = starflag(vtab['starflag'])
        visittab.write(visitfile,overwrite=True)
        #MWRFITS,visittab,visitfile,/silent
        allvisittab.append(visittab)
    # object loop
    logger.info(' aprv '+file_basename(planfile)+string(format='(f8.2)',systime(1)-t1)+string(format='(f8.2)',systime(1)-t0))


    # Save all RV info for all stars to apVisitSum file 
    #---------------------------
    hdu = fits.HDUList()
    # HDU0 - header only
    hdu.append(fits.ImageHDU())
    hdu[0].header['PLATEID'] = plantab['plateid']
    hdu[0].header['MJD'] = plantab['mjd']
    hdu[0].header['EXPTIME'] = header0['exptime'],'Total visit exptime per dither pos'
    hdu[0].header['JD-MID'] = header0['JD-MID'],' JD at midpoint of visit'
    hdu[0].header['UT-MID'] = header0['UT-MID'],' Date at midpoint of visit'
    ncombine = header0.get('NCOMBINE')
    if ncombine is None:
        ncombine = 1
    hdu[0].header['NCOMBINE'] = ncombine
    for j in range(ncombine):
        hdu[0].header['FRAME'+str(j+1)] = header0['FRAME'+str(j+1)],'Constituent frame'
    hdu[0].header['NPAIRS'] = header0['NPAIRS'],' Number of dither pairs combined'
    leadstr = 'AP1DVISIT: '
    hdu[0].header['V_APRED'] = plan.getgitvers(),'APOGEE software version'
    hdu[0].header['APRED'] = load.apred,'APOGEE Reduction version'
    hdu[0].header['HISTORY'] = leadstr+time.asctime()
    import socket
    hdu[0].header['HISTORY'] = leadstr+getpass.getuser()+' on '+socket.gethostname()
    import platform
    hdu[0].header['HISTORY'] = leadstr+'Python '+pyvers+' '+platform.system()+' '+platform.release()+' '+platform.architecture()[0]
    hdu[0].header['HISTORY'] = leadstr+load.prefix+'Visit information for '+str(len(allvisittab))+' Spectra'
    # HDU1 - table
    hdu.append(fits.table_to_hdu(allvisittab))
    hdu.writeto(visittabfile,overwrite=True)

    # Insert the apVisitSum information into the apogee_drp database
    if nobjind gt 0:
        print,'Loading visit data into the database'
        db = apogeedb.DBSession()
        db.ingest('visit',allvisittab)
        db.close()
        #DBINGEST_VISIT,allvisittab
    else:
        print('No data to load into the database')

    # plan files loop

    print('AP1DVISIT finished')
    logger.info('AP1DVISIT '+file_basename(planfile)+string(format='(f8.2)',systime(1)-t1)+string(format='(f8.2)',systime(1)-t0))
    dt = time.time()-t0
    print('dt = {:10.f} sec'.format(dt))

