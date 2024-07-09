import os
import time
import numpy as np
from astropy.io import fits
from astropy.table import Table
from ...utils import apload,plugmap,platedata,info


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
            plantab['field'] = apogee_field(0,planstr['plateid'])

	# Get APOGEE directories
        load = apload.ApLoad(apred=plantab['redux'],telescope=plantab['telescope'])
        #dirs = getdir(apogee_dir,cal_dir,spectro_dir,apred_vers=apred_vers,datadir=datadir)
        #logfile = apogee_filename('Diag',plate=planstr.plateid,mjd=planstr.mjd)
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
            plugmap = platedata.getdata(plantab['plateid'],planstr['mjd'],
                                        obj1m=plantab['apexp'][0]['singlename'],
                                        starfiber=planstr['apexp'][0]['single'],
                                        fixfiberid=fixfiberid)
        elif plantab['platetype']=='twilight':
            plugmap = platedata.getdata(plantab['plateid'],planstr['mjd'],twilight=True)
        elif plantab['platetype']=='cal':
            print('no plugmap for cal frames')
        else:
            plugfile = planstr['plugmap']
            plugmap = platedata.getdata(plantab['plateid'],planstr['mjd'],
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
        wavefiles = load.filename('Wave',chip=chiptag,num=planstr.waveid)
        # We are now using dailywave files with MJD names
        if plantab['waveid'] < 1e7:
            wavefiles = os.path.dirname(wavefiles[0])
            wavefiles += load.prefix+'Wave-'+chiptag+'-'+str(planstr['waveid'])+'.fits'
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
        nframes = len(planstr['apexp'])
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

            #------------------------------------------
            # Correcting and Calibrating the ap1D files
            #------------------------------------------
            cfiles = load.filename('Cframe',chip=chiptag,num=framenum,plate=plantab['plateid'],
                                   mjd=plantab['mjd'],field=plantab['field'])
            ctest = np.array([os.path.exists(f) for f in cfiles])
            if keyword_set(clobber) or $
            not file_test(cfiles[0]) or $
            not file_test(cfiles[1]) or $
            not file_test(cfiles[2]) then begin

            logger.info(' 1d processing '+file_basename(files[0])+string(format='(f8.2)',systime(1)-t1))
            
            # Load the 1D files
            #--------------------
            frame = load.frame(files)
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
                    remove_tags, chtab,'WCOEF',newtab
                    remove_tags, newtab,'WAVELENGTH',chtab
                # Add to the chip structure
                # Wavelength calibration data already added by ap2dproc with ap1dwavecal
                #if tag_exist(frame0.(0),'WCOEF') and not keyword_set(newwave) then begin
                if 'wcoef' in chtab.colnames: and newwave is None:
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
            print('STEP 1: Measuring the DITHER SHIFT with APDITHERSHIFT')
            # Not first frame, measure shift relative to 1st frame
            dither_commanded = sxpar(frame.(0).header,'DITHPIX')
            print('dither_commanded: ',dither_commanded)
            if j == 0:
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
                        pfile = plate_dir+'/plots/dithershift-'+framenum
                    else:
                        #pfile=0 & plot=0
                        plot = True
                        pfile = plate_dir+'/plots/dithershift-'+framenum
                if planstr['platetype']=='single':
                    nofit = True
                else:
                    nofit = False
                shiftout = dithershift(ref_frame,frame,xcorr=True,pfile=pfile,plot=plot,
                                       plugmap=plugmap,nofit=nofit,mjd=plantab['mjd'])
                shift = shiftout['shiftfit']
                shifterr = shiftout['shifterr']
                print('Measured dither shift: ',ashift,shift)
            # First frame, reference frame
            else:
                # measure shift anyway
                if j > 0:
                    shiftout = dithershift(ref_frame,frame,xcorr=True,pfile=pfile,plot=plot,
                                           plugmap=plugmap,nofit=nofit,mjd=plantab['mjd'])
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

            logger.info('  dithershift '+string(format='(f8.2)',systime(1)-t1)+string(format='(f8.2)',systime(1)-t0))

            if 'platetype'] in plantab.colnames:
                if plantab['platetype'] not in ['normal','single','twilight']:
                    continue

            #----------------------------------
            # STEP 2:  Wavelength Calibrate
            #----------------------------------
            # THIS IS NOW DONE AS PART OF AP2DPROC, USING PYTHON ROUTINES
            if ap1dwavecal:
                print('STEP 2: Wavelength Calibrating with AP1DWAVECAL')
                plotfile = plate_dir+'/plots/pixshift_chip-'+framenum
                if dithonly:
                    ap1dwavecal_refit(frame,frame_wave,plugmap=plugmap,verbose=True,plot=True,pfile=plotfile)
                plotfile = plate_dir+'/plots/pixshift-'+framenum
                if planstr['platetype'] == 'twilight':
                    ap1dwavecal(frame_shift,frame_wave,verbose=True,plot=True,pfile=plotfile)
                else:
                    ap1dwavecal(frame_shift,frame_wave,plugmap=plugmap,verbose=True,plot=True,pfile=plotfile)
                del frame  # free up memory
                logger.info('  wavecal '+string(format='(f8.2)',systime(1)-t1)+string(format='(f8.2)',systime(1)-t0))
            else:
                frame_wave = frame_shift

            #----------------------------------
            # STEP 3:  Airglow Subtraction
            #----------------------------------
            print('STEP 3: Airglow Subtraction with APSKYSUB')
            sky.skysub(frame_wave,plugmap,frame_skysub,subopt=1,error=skyerror,force=force)
            #if n_elements(skyerror) gt 0 and planstr.platetype ne 'twilight' then begin
            #  stop,'halt: APSKYSUB Error: ',skyerror
            #  apgundef,frame_wave,frame_skysub,skyerror
            #endif
            del frame_wave  # free up memory
            logger.info('  airglow '+string(format='(f8.2)',systime(1)-t1)+string(format='(f8.2)',systime(1)-t0))

            if 'platetype' in plantab.colnames:
                if plantab['platetype'] not in ['normal','single','twilight']:
                    continue

            #----------------------------------
            # STEP 4:  Telluric Correction
            #----------------------------------
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
                visitstr = 0
            if 'pltelstarfit'] in plantab.colnames:
                pltelstarfit = planstr['pltelstarfit']
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
                         plots_dir=plots_dir,error=telerror,/save,/preconv,visitstr=visitstr,
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
            logger.info(logfile,'  telluric '+string(format='(f8.2)',systime(1)-t1)+string(format='(f8.2)',systime(1)-t0))

            #-----------------------
            # Output apCframe files
            #-----------------------
            print('Writing output apCframe files')
            outfiles = load.filename('Cframe',chip=chiptag,num=framenum,
                                     plate=plantab['plateid'],mjd=plantab['mjd'],
                                     field=plantab['field'])
            outcframe(frame_telluric,plugmap,outfiles,verbose=False)

        # frame loop done

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

        # Update SHIFTSTR
        shiftstr[j]['index'] = j
        shiftstr[j]['framenum'] = framenum
        shiftstr[j]['shift'] = shift
        shiftstr[j]['shifterr'] = shifterr
        shiftstr[j]['pixshift'] = pixshift
        shiftstr[j]['shiftfit'] = frame_telluric['shift']['shiftfit']
        shiftstr[j]['chipshift'] = frame_telluric['shift']['chipshift']
        shiftstr[j]['chipfit'] = frame_telluric['shift']['chipfit']
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
        shiftstr[j]['sn'] = np.median(frame_telluric[1]['flux'][:,fbright]/frame_telluric[1].['err'][:,fbright])

