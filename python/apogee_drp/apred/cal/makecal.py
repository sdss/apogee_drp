import os
import subprocess
import numpy as np
from astropy.io import fits
from scipy.signal import medfilt2d


def makecal(indexfile,name,caltype,**kwargs):
#def makecal(indexfile,det=None,dark=None,flat=None,wave=None,multiwave=None,
#            lsf=None,bpm=None,psf=None,flux=None,sparse=None,fiber=None,
#            littrow=None,persist=None,modelpersist=None,
#            detid=None,darkid=None,flatid=None,waveid=None,lsfid=None,
#            response=None,mjd=None,full=False,newwave=None,nskip=None,
#            average=False,clobber=False,apred='',telescope='',
#            nofit=False,doplot=False,unlock=False,fpi=None,librarypsf=False,
#            dailywave=None,telluric=None,modelpsf=None):
    """
    This will make one or ALL of the specified calibration product types
    listed in the master calibration index file.

    Parameters
    ----------
    indexfile : str
        Name of master calibration index file, if not
         specified use default cal.par in calibration directory
    dark : 
       Make all of the darks in the file
    dark=darkid
       Make the dark with name=darkid 
    flat         
       Make all of the flats in the file
    flat=flatid
       Make the flat with name=flatid 
    wave
       Make all of the wavecals in the file
    wave : int
       Make the wavecal with name=waveid .
    lsf          
       Make all of the lsfs in the file
    lsf : int
       Make the lsf with name=lsfid.
    fpi : int
       Make the FPI with name=fpiid.
    librarypsf : bool, optional
       Use PSF library to get PSF cal for images.  Default is 
    dailywave : int
       Daily wavelength solution ID.
    modelpsf : int
       Model PSF ID.

    Returns
    -------

    Calibration products are generated in places specified by the
    SDSS/APOGEE directory tree.

    Examples
    --------
    
    makecal(file=file,/dark,/flat,/wave,/lsf)
        OR
    makecal(file=file,dark=darkid,flat=flatid,wave=waveid,lsf=lsfid)
    
    Written by J.Holtzman Aug 2011
    Added doc strings and general cleanup by D. Nidever, Sep 2020
    Translated to Python  D. Nidever  2023/2024
    """

    images = np.atleast_id(waveid)
    if name is None:
        name = str(image[0])

    load = apload.ApLoad(apred=apred,telescope=telescope)


    if keyword_set(vers) and keyword_set(telescope) then apsetver,vers=vers,telescope=telescope
    dirs = getdir(apo_dir,cal_dir,spectro_dir,apo_vers,lib_dir)

    # Get default file name if file not specified
    if keyword_set(indexfile):
        if strpos(indexfile,'/') lt 0 then indexfile=file_dirname(dirs.calfile)+'/'+indexfile 
    else:
        indexfile = load.calfile
    calfile = dirs.calfile

    if not keyword_set(full) then full=0
    if not keyword_set(newwave) then newwave=0
    if not keyword_set(nskip) then nskip=1
    chips = ['a','b','c']

    # Read calibration master file into calibration structures
    caltab = readcal(indexfile)
    #,darkstr,flatstr,sparsestr,fiberstr,badfiberstr,fixfiberstr,wavestr,lsfstr,bpmstr,$
    #        fluxstr,detstr,littrowstr,persiststr,persistmodelstr,responsestr,multiwavestr,modelpsfstr

    funcdict = {'det':mkdet,'dark':mkdark,'flat':mkflat,'bpm':mkbpm,'sparse':mksparse,
                'fiber':mkfiber,'psf':mkpsf,'modelpsf':mkmodelpsf,'fpi':mkfpi,
                'littrow':mklittrow,'persist':mkpersist,'persistmodel':mkpersistmodel,
                'flux':mkflux,'response':mkresponse,'wave':mkwave,'multiwave':mkmultiwave,
                'dailywave':mkdailywave,'telluric':mktelluric,'lsf':mklsf}

    if caltype not in functdict.keys():
        raise ValueError(str(caltype)+' not supported')
    
    # Call the appropriate function
    functdict[caltype](name,**kwargs)
    
    # mkdet(**kwargs)
    # mkdark(**kwargs)
    # mkflat(**kwargs)
    # mkbpm(**kwargs)
    # mksparse(**kwargs)
    # mkfiber(**kwargs)
    # mkpsf(**kwargs)
    # mkmodelpsf(**kwargs)
    # mkfpi(**kwargs)
    # mklittrow(**kwargs)
    # mkpersist(**kwargs)
    # mkpersistmodel(**kwargs)
    # mkflux(**kwargs)
    # mkresponse(**kwargs)
    # mkwave(**kwargs)
    # mkmultiwave(**kwargs)
    # mkdailywave(**kwargs)
    # mktelluric(**kwargs)
    # mklsf(**kwargs)

    
def mkdet(name,**kwargs):
    """
    Make Detector calibration files
    """
    print('makecal det:',name)
    outfile = load.filename('Detector',num=name,chips=True)
    outdir = os.path.dirname(load.filename('Detector',num=name,chips=True))
    allfiles = os.path.join(detdir,load.prefix+'Detector-{:s}-{:08d}.fits'.format(chips,sdetid))
    if np.sum(np.array([os.path.exists(fil) for fil in allfiles]))==3 and clobber==False:
        print(' detector file: ',outfile,' already made')
        return
    i = where(detstr.name == name)
    if i>0:
        print('No matching calibration line for',name)
    mkdet(detstr[i]['name'],detstr[i]['linid'],unlock=kwargs['unlock'])

def mkdark(name,**kwargs):
    """
    Make Dark calibration files
    """
    print('makecal dark:',name)
    darkfile = load.filename('Dark',num=name,chips=True)
    darkdir = os.path.dirname(load.filename('Dark',num=name,chips=True))
    allfiles = darkdir+'/'+[load.prefix+'Dark-'+chips+'-'+sdarkid+'.fits',dirs.prefix+'Dark-'+sdarkid+'.tab']
    if np.sum(file_test(allfiles)) == 4 and clobber==False:
        print(' dark file: ',darkfile+'.tab',' already made')
        return
    
    i, = np.where(darkstr.name == name)
    if i < 0:
        print('No matching calibration line for',name)

      ims = getnums(darkstr[i].frames)
      cmjd = getcmjd(ims[0],mjd=mjd)
      getcal(mjd,calfile,detid=detid)
      makecal(det=detid,unlock=unlock)
      mkdark(ims,clobber=clobber,unlock=unlock)
    else:
      if keyword_set(mjd):
          num = getnum(mjd) 
          red, = np.where(darkstr.frames/10000L == num)
      else:
          red = np.arange(len(darktab))
      if (red[0] >= 0):
          for i in range(len(red)):
              ims = getnums(darkstr[red[i]].frames)
              cmjd = getcmjd(ims[0],mjd=mjd)
              getcal(mjd,calfile,detid=detid)
              makecal(det=detid,unlock=unlock)
              mkdark(ims,clobber=clobber,unlock=unlock)

def mkflat(name,**kwargs):
    """
    Make Flat calibration files
    """
    print('makecal flat:',name)
    flatfile = load.filename('Flat',num=name,chips=True)
    flatdir = os.path.dirname(load.filename('Flat',num=name,chips=True))
    allfiles = flatdir+dirs.prefix+'Flat-'+chips+'-'+sflatid+'.fits'
    allfiles = [allfiles,flatdir+dirs.prefix+'Flat-'+sflatid+'.tab']
    if np.sum(file_test(allfiles)) == 4 and clobber==False:
        print(' flat file: ',flatfile+'.tab',' already made')
        return

    i, = np.where(flatstr.name == name)
    if i <0:
        print('No matching calibration line for',name)
      ims = getnums(flatstr[i].frames)
      cmjd = getcmjd(ims[0],mjd=mjd)
      getcal(mjd,calfile,darkid=darkid)
      makecal(dark=darkid,unlock=unlock)
      mkflat(ims,darkid=darkid,nrep=flatstr[i].nrep,dithered=flatstr[i].dithered,
             clobber=clobber,unlock=unlock)
    else:
      if keyword_set(mjd):
          num = getnum(mjd) 
          red, = np.where(flatstr.frames/10000L == num)
      else:
          red = np.arange(len(darktab))
      if (red[0] >= 0):
            for i in range(len(red)):
                ims = getnums(flatstr[red[i]].frames)
                cmjd = getcmjd(ims[0],mjd=mjd)
                getcal(getcal(mjd,calfile,darkid=darkid))
                makecal(dark=darkid,unlock=unlock)
                mkflat(ims,darkid=darkid,nrep=flatstr[i].nrep,dithered=flatstr[i].dithered,
                       clobber=clobber,unlock=unlock)

def mkbpm(name,**kwargs):
    """
    Make BPM calibration files
    """
    print('makecal bpm:',name)
    bpmfile = load.filename('BPM',num=name,chips=True)
    bpmdir = os.path.dirname(load.filename('BPM',num=name,chips=True))
    allfiles = bpmdir+dirs.prefix+'BPM-'+chips+'-'+sbpmid+'.fits'
    if np.sum(file_test(allfiles)) == 3 and clobber==False:
        print(' bpm file: ',bpmfile, ' already made')
        return
    i = where(bpmstr.name == name)
    if i < 0:
        print('No matching calibration line for',name)
    makecal(dark=bpmstr[i].darkid,unlock=unlock)
    makecal(flat=bpmstr[i].flatid,unlock=unlock)
    mkbpm(bpmstr[i].name,darkid=bpmstr[i].darkid,flatid=bpmstr[i].flatid,
          clobber=clobber,unlock=unlock)
    else:
      if keyword_set(mjd):
          num = getnum(mjd) 
          red, = np.where(bpmstr.frames/10000L == num)
      else:
          red = np.arange(len(bpmtab))
      if (red[0] >= 0):
        for i in range(len(red)):
            makecal(dark=bpmstr[i].darkid,clobber=clobber,unlock=unlock)
            makecal(flat=bpmstr[i].flatid,clobber=clobber,unlock=unlock)
            mkbpm(bpmstr[red[i]].name,darkid=bpmstr[i].darkid,flatid=bpmstr[i].flatid,
                  clobber=clobber,unlock=unlock)

def mksparse(name,**kwargs):
    """
    Make Sparsepak PSF calibration product
    """
    print('makecal sparse:',name)
    sparsefile = load.filename('Sparse',num=name,chips=True)
    psfdir = os.path.dirname(load.filename('Sparse',num=name,chips=True))
    if os.path.exists(sparsefile) and clobber==False:
        print(' sparse file: ',sparsefile,' already made')
        return
    i = where(sparsestr.name == name)
    if i < 0:
        print('No matching calibration line for',name)
    ims = getnums(sparsestr[i].frames)
    cmjd = getcmjd(ims[0],mjd=mjd)
    getcal(mjd,calfile,darkid=darkid,flatid=flatid,bpmid=bpmid)
    makecal(dark=darkid,unlock=unlock)
    makecal(flat=flatid,unlock=unlock)
    makecal(bpm=bpmid,unlock=unlock)
    darkims = getnums(sparsestr[i].darkframes)
    maxread = getnums(sparsestr[i].maxread)
    if n_len(maxread) != 3:
          print('sparse maxread does not have 3 elements! ')
       mkepsf(ims,darkid=darkid,flatid=flatid,darkims=darkims,dmax=sparsestr[i].dmax,
             maxread=maxread,clobber=clobber,/filter,thresh=0.2,scat=2,unlock=unlock)
      # This creates apSparse and apEPSF files
      # Make empty apPSF files to indicate to makecal.pro that this
      #  PSF file was already made
      ssparse = string(sparse,format='(i08)')
      psffiles = psfdir+'/'+[dirs.prefix+'PSF-'+chips+'-'+ssparse+'.fits']  
      touchzero(psffiles)


def mkfiber(name,**kwargs):
    """
    Make fiber calibration file
    """
    print('makecal fiber:',name)
    psffile = load.filename('PSF',num=name,chips=True)
    psfdir = os.path.dirname(load.filename('PSF',num=name,chips=True))
    allfiles = psfdir+'/'+[dirs.prefix+'EPSF-'+chips+'-'+sfiberid+'.fits',dirs.prefix+'PSF-'+chips+'-'+sfiberid+'.fits']
    if np.sum(file_test(allfiles)) == 6 and clobber==False:
        print(' psf file: ',file, ' already made')
        return
      cmjd = getcmjd(fiber,mjd=mjd)
      getcal(mjd,calfile,darkid=darkid,flatid=flatid,sparseid=sparseid)
      mkpsf(fiber,darkid=darkid,flatid=flatid,sparseid=sparseid,unlock=unlock)

def mkpsf(name,**kwargs):
    """
    Make PSF calibration file
    """
    #if keyword_set(psf) and not keyword_set(flux) and not keyword_set(wave)
    print('makecal psf:',name)
    psffile = load.filename('PSF',num=name,chips=True)
    psfdir = os.path.dirname(load.filename('PSF',num=name,chips=True))
    allfiles = psfdir+'/'+[dirs.prefix+'EPSF-'+chips+'-'+spsfid+'.fits',dirs.prefix+'PSF-'+chips+'-'+spsfid+'.fits']
    if np.sum(file_test(allfiles)) == 6 and clobber==False:
        print(' psf file: ',file, ' already made')
        return
      cmjd = getcmjd(psf,mjd=mjd)
      getcal(mjd,calfile,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,
             littrowid=littrowid,bpmid=bpmid)
      makecal(littrow=littrowid,unlock=unlock)
      mkpsf(psf,bpmid=bpmid,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,
            littrowid=littrowid,clobber=clobber,unlock=unlock)

def mkmodelpsf(name,**kwargs):
    """
    Make Model PSF calibration file
    """
    if keyword_set(modelpsf) and (not keyword_set(fpi) and not keyword_set(flux) and not keyword_set(wave)) then begin
    print('makecal modelpsf:',name)
    if modelpsf gt 1 or size(modelpsf,/type) eq 7 then begin 
    psffile = load.filename('PSFModel',num=name,chips=True)
    psfdir = os.path.dirname(load.filename('PSFModel',num=name,chips=True))
    #spsfid = string(modelpsf,format='(i08)')
    allfiles = psfdir+'/'+dirs.prefix+'PSFModel-'+chips+'-'+spsfid+'.fits'
    if np.sum(file_test(allfiles)) == 3 and clobber==False:
        print(' modelpsf file: ',psffile, ' already made')
        return
      i = where(modelpsfstr.name == modelpsf)
      if i < 0:
          print('No matching calibration line for',modelpsf)

      makecal(sparse=modelpsfstr[i].sparse,unlock=unlock)
      makecal(psf=modelpsfstr[i].psf,unlock=unlock)
      mkmodelpsf(modelpsf,sparseid=modelpsfstr[i].sparse,psfid=modelpsfstr[i].psf,clobber=clobber,unlock=unlock)

def mkfpi(name,**kwargs):
    """
    Make FPI calibration file
    """
    print('makecal fpi:',name)
    wavefpifile = load.filename('WaveFPI',num=name,chips=True)
    wavefpidir = os.path.dirname(load.filename('WaveFPI',num=name,chips=True))
    wavedir = file_dirname(file)
    sfpiid = string(fpi,format='(i08)')
    cmjd = getcmjd(fpi[0],mjd=mjd)
    allfiles = wavedir+'/'+dirs.prefix+'WaveFPI-'+chips+'-'+cmjd+'-'+sfpiid+'.fits'
    if np.sum(file_test(allfiles)) == 3 and clobber==False:
        print,' fpi file: ',file, ' already made'
        return
      getcal(mjd,calfile,darkid=darkid,flatid=flatid,bpmid=bpmid,fiberid=fiberid,modelpsf=psfmodel)
      # Use Model PSF by default
      if not keyword_set(psf) and not keyword_set(librarypsf):
          psfid = 0
          modelpsf = psfmodel
          makecal(modelpsf=modelpsf,unlock=unlock)
      # Use PSF file     
      else:
        # What PSF to use
        if keyword_set(psf):
            psfid = psf
        # Try to find a PSF from this day
        else:
            print,'Trying to automatically find a PSF calibration file'
            psfid = getpsfcal(fpi[0],psflibrary=librarypsf,unlock=unlock)
        makecal(psf=psfid,unlock=unlock)
      makecal(fiber=fiberid,unlock=unlock)
      makecal(dailywave=mjd,unlock=unlock,librarypsf=librarypsf,modelpsf=modelpsf)
      mkfpi(fpi,name=name,darkid=darkid,flatid=flatid,psfid=psfid,$0
            fiberid=fiberid,clobber=clobber,unlock=unlock,psflibrary=librarypsf,modelpsf=modelpsf)

def mklittrow(name,**kwargs):
    """
    Make Littrow calibration file
    """
    print('makecal littrow:',name)
    litfile = load.filename('Littrow',num=name,chips=True)
    if os.path.exists(litfile) and clobber==False:
        print(' littrow file: ',litfile,' already made')
        return
    cmjd = getcmjd(littrow,mjd=mjd)
    getcal(mjd,calfile,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid)
    makecal(flat=flatid,unlock=unlock)
    mklittrow(littrow,cmjd=cmjd,darkid=darkid,flatid=flatid,sparseid=sparseid,
              fiberid=fiberid,clobber=clobber,unlock=unlock)

def mkpersist(name,**kwargs):
    """
    Make Persistence calibration file
    """
    print('makecal persist:',name)
    perfile = load.filename('Persist',num=name,chips=True)
    perdir = os.path.dirname(load.filename('Persist',num=name,chips=True))
    allfiles = perdir+'/'+dirs.prefix+'Persist-'+chips+'-'+sperid+'.fits'
    if np.sum(file_test(allfiles)) == 3 and clobber==False:
        print(' persist file: ',perfile, ' already made')
        return
    i = where(persiststr.name == persist)
    if i lt 0:
        print('No matching calibration line for',persist)
    cmjd = getcmjd(persist,mjd=mjd)
    getcal(mjd,calfile,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid)
    mkpersist(persist,persiststr[i].darkid,persiststr[i].flatid,thresh=persiststr[i].thresh,
              cmjd=cmjd,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,
              clobber=clobber,unlock=unlock)

def mkpersistmodel(name,**kwargs):
    """
    Make Persistence model calibration file
    """
    print('makecal modelpersist:',name)
    perfile = load.filename('PersistModel',num=name,chips=True)
    perdir = os.path.dirname(load.filename('PersistModel',num=name,chips=True))
    allfiles = perdir+'/'+dirs.prefix+'PersistModel-'+chips+'-'+sperid+'.fits'
    if np.sum(file_test(allfiles)) eq 3 and clobber==False:
        print(' modelpersist file: ',file, ' already made')
        return
    i = where(persistmodelstr.name == modelpersist)
    if i<0:
        print('No matching calibration line for',modelpersist)
    cmjd = getcmjd(modelpersist,mjd=mjd)
    getcal(mjd,calfile,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid)
    mkpersistmodel(modelpersist)

def mkflux(name,**kwargs):
    """
    Make Flux calibration file
    """
    print('makecal flux:',name)
    fluxfile = load.filename('Flux',num=name,chips=True)
    fluxdir = os.path.dirname(load.filename('Flux',num=name,chips=True))
    allfiles = fluxdir+'/'+load.prefix+'Flux-'+chips+'-'+sfluxid+'.fits'      
    if np.sum(file_test(allfiles)) == 3 and clobber==False:
        print(' flux file: ',fluxfile, ' already made')
        return
    cmjd = getcmjd(flux[0],mjd=mjd)
    getcal(mjd,calfile,darkid=darkid,flatid=flatid,littrowid=littrowid,waveid=waveid,modelpsf=psfmodel)
    # Use Model PSF by default
    if not keyword_set(psf) and not keyword_set(librarypsf) and keyword_set(psfmodel):
        psfid = 0
        modelpsf = psfmodel
        makecal(modelpsf=modelpsf,unlock=unlock)
    # Use PSF file     
    else:
        if keyword_set(psf):
            psfid = psf
        # Try to find a PSF from this day
        else:
            print('Trying to automatically find a PSF calibration file')
            psfid = getpsfcal(flux[0],psflibrary=librarypsf,unlock=unlock)
        makecal(psf=psfid,unlock=unlock)
      makecal(littrow=littrowid,unlock=unlock)
      mkflux(flux,darkid=darkid,flatid=flatid,psfid=psfid,modelpsf=modelpsf,littrowid=littrowid,
             waveid=waveid,clobber=clobber,unlock=unlock)

def mkresponse(name,**kwargs):
    """
    Make Response calibration file
    """
    print('makecal response:',name)
    resfile = load.filename('Response',num=name,chips=True)
    resdir = os.path.dirname(load.filename('Response',num=name,chips=True))
    allfiles = resdir+'/'+dirs.prefix+'Response-'+chips+'-'+sresid+'.fits'
    if np.sum(file_test(allfiles)) == 3 and clobber==False:
        print(' response file: ',resfile, ' already made')
        return
    i = where(responsestr.name eq response,nres)
    if nres == 0:
        print('No matching calibration line for ', response)
    else:
        nres>1:
            i=i[0]
    cmjd = getcmjd(response,mjd=mjd)
    getcal(mjd,calfile,darkid=darkid,flatid=flatid,littrowid=littrowid,waveid=waveid,fiberid=fiberid)
    makecal(psf=responsestr[i].psf,unlock=unlock)
    makecal(wave=waveid,unlock=unlock)
    makecal(fiber=fiberid,unlock=unlock)
    makecal(littrow=littrowid,unlock=unlock)
    mkflux(response,darkid=darkid,flatid=flatid,psfid=responsestr[i].psf,littrowid=littrowid,
           waveid=waveid,temp=responsestr[i].temp,clobber=clobber,unlock=unlock)

def mkwave(name,**kwargs):
    """
    Make Wavelength calibration file
    """
    print('makecal wave:',name)
    wavefile = load.filename('Wave',num=name,chip='c')
    wavedir = os.path.dirname(load.filename('Wave',num=name,chip='c'))
    swaveid = string(wave,format='(i08)')
    allfiles = wavedir+'/'+dirs.prefix+'Wave-'+chips+'-'+swaveid+'.fits'
    if np.sum(file_test(allfiles)) == 3 and clobber==False:
        print(' wave file: ',wavefile, ' already made')
        return
    i = where(wavestr.name eq wave,nwave)
    if nwave > 0:
        ims = getnums(wavestr[i[0]].frames)
        name = wavestr[i[0]].name
        psfid = wavestr[i[0]].psfid
     # Use the input filename
     else:
        ims = wave
        name = ims[0]
        cmjd = getcmjd(ims[0],mjd=mjd)
        getcal(mjd,calfile,modelpsf=psfmodel)
        # Use Model PSF by default
        if not keyword_set(psf) and not keyword_set(librarypsf):
            psfid = 0
            modelpsf = psfmodel
            makecal(modelpsf=modelpsf,unlock=unlock)
        # Use PSF file     
        else:
          if keyword_set(psf):
              psfid = psf
          # Try to find a PSF from this day
          else:
              print('Trying to automatically find a PSF calibration file')
              psfid = getpsfcal(ims[0],psflibrary=librarypsf,unlock=unlock)
          makecal(psf=psfid,unlock=unlock)
      cmjd = getcmjd(ims[0],mjd=mjd)
      getcal(mjd,calfile,darkid=darkid,flatid=flatid,bpmid=bpmid,fiberid=fiberid)
      makecal(bpm=bpmid,unlock=unlock)
      makecal(fiber=fiberid,unlock=unlock)
      mkwave(ims,name=name,darkid=darkid,flatid=flatid,psfid=psfid,modelpsf=modelpsf,
             fiberid=fiberid,clobber=clobber,nofit=nofit,unlock=unlock)
    else:
      if keyword_set(mjd):
          num = getnum(mjd) 
          red = where(wavestr.frames/10000L eq num)
      else:
          red = np.arange(len(wavetab))
      if (red[0] >= 0):
          for i in np.arange(0,len(red),nskip):
              ims = getnums(wavestr[red[i]].frames)
              cmjd = getcmjd(ims[0],mjd=mjd)
              getcal(mjd,calfile,darkid=darkid,flatid=flatid,bpmid=bpmid,fiberid=fiberid)
              makecal(bpm=bpmid,unlock=unlock)
              makecal(fiber=fiberid,unlock=unlock)
              mkwave(ims,name=wavestr[red[i]].name,darkid=darkid,flatid=flatid,psfid=wavestr[red[i]].psfid,
                     fiberid=fiberid,clobber=clobber,/nowait,nofit=nofit,unlock=unlock)
            
def mkmultiwave(name,**kwargs):
    """
    Make multi-night wavelength calibration file
    """
    print('makecal multiwave:',name)
    wavefile = load.filename('Wave',num=name,chips=True)
    wavedir = os.path.dirname(load.filename('Wave',num=name,chips=True))
    allfiles = wavedir+dirs.prefix+'Wave-'+chips+'-'+swaveid+'.fits'
    allfiles = [allfiles,wavedir+dirs.prefix+'Wave-'+swaveid+'py.dat']
    if np.sum(file_test(allfiles)) == 4 and clobber==False:
        print(' multiwave file: ',wavefile+'.dat',' already made')
        return
    i = where(multiwavestr.name eq multiwave,nwave)
    if nwave le 0:
        print,'No matching calibration line for ', multiwave
    ims = getnums(multiwavestr[i[0]].frames)
    mkmultiwave(ims,name=multiwavestr[i[0]].name,clobber=clobber,file=file,unlock=unlock,
                psflibrary=librarypsf)
    else:
      if keyword_set(mjd):
          num = getnum(mjd) 
          red = where(multiwavestr.frames/10000L eq num)
      else:
          red = np.arange(len(multiwavetab))
      if (red[0] >= 0):
          for i in np.arange(0,len(red),nskip):
              ims = getnums(multiwavestr[red[i]].frames)
              mkmultiwave(ims,name=multiwavestr[red[i]].name,clobber=clobber,file=file,unlock=unlock,
                          nowait=True,psflibrary=librarypsf)

def mkdailywave(name,**kwargs):
    """
    Make daily wavelength calibration file
    """
    print('makecal dailywave:',name)
    dir = load.filename('Wave',num=name,/nochip,/dir)
    wavefile = dir+dirs.prefix+'Wave-'+str(name)+'.fits'
    swaveid = strtrim(dailywave,2)
    allfiles = dir+dirs.prefix+'Wave-'+chips+'-'+swaveid+'.fits'
    if np.sum(file_test(allfiles)) == 3 and clobber==False:
        print(' dailywave file: ',wavefile,' already made')
        return
    mjd = dailywave
    getcal(mjd,calfile,darkid=darkid,flatid=flatid,bpmid=bpmid,fiberid=fiberid,modelpsf=modelpsf)
    #if keyword_set(librarypsf) then apgundef,modelpsf
    makecal(bpm=bpmid,unlock=unlock)
    makecal(fiber=fiberid,unlock=unlock)
    mkdailywave(dailywave,darkid=darkid,flatid=flatid,psfid=psfid,
                fiberid=fiberid,clobber=clobber,nofit=nofit,unlock=unlock,
                psflibrary=librarypsf,modelpsf=modelpsf)

def mktelluric(name,**kwargs):
    """
    Make daily telluric calibration file
    """
    print('makecal telluric:',name)
    teldir = load.filename('Telluric',num=name,/nochip,/dir)
    telfile = dir+dirs.prefix+'Telluric-'+str(name)+'.fits'
    allfiles = dir+dirs.prefix+'Telluric-'+chips+'-'+telluric+'.fits'
    allfiles = [allfiles,dir+dirs.prefix+'Telluric-'+telluric+'.dat']
    if np.sum(file_test(allfiles)) == 4 and clobber==False:
        print(' telluric file: ',telfile,' already made')
        return
    waveid = int((strsplit(telluric,'-',/extract))[0])
    lsfid = int((strsplit(telluric,'-',/extract))[1])
    if waveid < 1e7:
       makecal(dailywave=waveid,unlock=unlock)
    else:
       makecal(wave=waveid,unlock=unlock)
    makecal(lsf=lsfid,unlock=unlock)
    mktelluric(telluric,clobber=clobber,unlock=unlock)

def mklsf(name,**kwargs):
    """
    Make LSF calibration file
    """
    print('makecal lsf:',name)
    lsffile = load.filename('LSF',num=name,chips=True)
    slsfid = string(lsf,format='(i08)')
    lsfdir = os.path.dirname(load.filename('LSF',num=name,chips=True))
    allfiles = lsfdir+dirs.prefix+'LSF-'+chips+'-'+slsfid+'.fits'
    allfiles = [allfiles,lsfdir+dirs.prefix+'LSF-'+slsfid+'.sav']
    if np.sum(file_test(allfiles)) == 4 and clobber==False:
        print(' lsf file: ',file+'.sav',' already made')
        return
    i, = np.where(lsfstr.name eq lsf,nlsf)
    if nlsf <= 0:
        print('No matching calibration line for',lsf)
    ims = getnums(lsfstr[i[0]].frames)
    cmjd = getcmjd(ims[0],mjd=mjd)
    getcal(mjd,calfile,darkid=darkid,flatid=flatid,multiwaveid=waveid,fiberid=fiberid)
    makecal(multiwave=waveid,unlock=unlock)
    mklsf(ims,waveid,darkid=darkid,flatid=flatid,psfid=lsfstr[i[0]].psfid,fiberid=fiberid,
          full=full,newwave=newwave,clobber=clobber,pl=pl,unlock=unlock)
    else:
      if keyword_set(mjd):
        num = getnum(mjd) 
        red = where(lsfstr.frames/10000L eq num)
      else:
          red = np.arnage(len(lsftab))
      if (red[0] >= 0):
        for i in np.arange(len(red),nskip):
            ims = getnums(lsfstr[red[i]].frames)
            cmjd = getcmjd(ims[0],mjd=mjd)
            getcal(mjd,calfile,darkid=darkid,flatid=flatid,multiwaveid=waveid,fiberid=fiberid,modelpsf=modelpsf)
            if keyword_set(librarypsf) then apgundef,modelpsf
            makecal(multiwave=waveid,unlock=unlock,librarypsf=librarypsf,modelpsf=modelpsf)
            print('calling mklsf')
            mklsf(ims,waveid,darkid=darkid,flatid=flatid,psfid=lsfstr[i].psfid,fiberid=fiberid,
                  full=full,newwave=newwave,clobber=clobber,pl=pl,unlock=unlock,nowait=True)
