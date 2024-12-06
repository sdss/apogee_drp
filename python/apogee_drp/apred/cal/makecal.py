import os
import subprocess
import numpy as np
from astropy.io import fits
from scipy.signal import medfilt2d
from ..mkcal import getcal,readcal
from ...utils import apload
from .. import cal

chips = ['a','b','c']

def makecal(name,caltype,apred=None,telescope=None,**kw):
    """
    This will make one or ALL of the specified calibration product types
    listed in the master calibration index file.

    Parameters
    ----------
    name : str
       ID or name of the calibration file.
    caltype : str
       Name of the calibration, e.g. "det", "dark", "wave".
         det, dark, flat, bpm, sparse, fiber, psf, modelpsf,
         fpi, littrow, persist, persistmodel, flux, 
         response, wave, multiwave, dailywave, telluric, lsf
    apred : str, optional
       APOGEE reduction version, e.g. "daily" or "1.4".
    telescope :str, optional
       Telescope name, either "apo25m" or "lco25m".
    load : ApLoad object, optional
       ApLoad object with reduction information.  Either this or 
         apred+telescope have to be input.
    calfile : str, optional
        Name of master calibration index file, if not
         specified use default cal.par in calibration directory

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

    if 'load' in kw.keys():
        load = kw['load']
    elif apred is not None and telescope is not None:
        load = apload.ApLoad(apred=apred,telescope=telescope)
        kw['load'] = load
    else:
        raise ValueError('Either apred+telescope or load have to be input')
        
    # Get default file name if file not specified
    if 'calfile' not in kw.keys() or kw['calfile'] is None or kw['calfile']=='':
        caldir = os.path.join(os.environ['APOGEE_DRP_DIR'],'data','cal')
        calfile = os.path.join(caldir,load.instrument+'.par')
    else:
        calfile = kw['calfile']
    kw['calfile'] = calfile
    # Get calibration dictionary
    if 'allcaldict' not in kw.keys():
        allcaldict = readcal(calfile)
        kw['allcaldict'] = allcaldict
    if 'verbose' not in kw.keys():
        kw['verbose'] = False
    
    funcdict = {'det':det,'dark':dark,'flat':flat,'bpm':bpm,'sparse':sparse,
                'fiber':fiber,'psf':psf,'modelpsf':modelpsf,'fpi':fpi,
                'littrow':littrow,'persist':persist,'persistmodel':persistmodel,
                'flux':flux,'response':response,'wave':wave,'multiwave':multiwave,
                'dailywave':dailywave,'telluric':telluric,'lsf':lsf}

    if caltype not in functdict.keys():
        raise ValueError(str(caltype)+' not supported')

    # Call the appropriate function
    if kw['verbose']:
        print('makecal '+str(caltype)+': '+str(name))
    functdict[caltype](name,**kw)

    
def det(name,**kw):
    """ Make Detector calibration files """
    load = kw['load']
    detfile = load.filename('Detector',num=name,chips=True)
    detdir = os.path.dirname(load.filename('Detector',num=name,chips=True))
    fmt = '{:s}Detector-{:s}-{:08d}.fits'
    outfiles = [os.path.join(detdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles]))==3 and clobber==False:
        print(' detector file: ',detfile,' already made')
        return
    dettab = kw['allcaldict']['det'] 
    ind, = np.where(dettab['name']==str(name))
    if len(ind)==0:
        print('No matching calibration line for',name)
        return
    else:
        ind = ind[0]
    cal.mkdet(dettab[ind]['name'],dettab[ind]['linid'],unlock=kw['unlock'])

def dark(name,**kw):
    """ Make Dark calibration files """
    load = kw['load']
    darkfile = load.filename('Dark',num=name,chips=True)
    darkdir = os.path.dirname(load.filename('Dark',num=name,chips=True))
    fmt = '{:s}Dark-{:s}-{:08d}.fits'
    outfiles = [os.path.join(darkdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    outfiles += [os.path.join(darkdir,load.prefix+'Dark-{:08d}.tab'.format(int(name)))]
    if np.sum([os.path.exists(f) for f in outfiles])==4 and clobber==False:
        print(' dark file: ',darkfile+'.tab',' already made')
        return
    darktab = kw['allcaldict']['dark']
    ind, = np.where(darktab['name']==str(name))
    if len(ind)==0:
        print('No matching calibration line for',name)
        return
    else:
        ind = ind[0]
    ims = getnums(darktab[ind]['frames'])
    mjd = int(load.cmjd(ims[0]))
    caldict = getcal(kw['calfile'],mjd)
    makecal(det=detid,unlock=kw['unlock'])
    cal.mkdark(ims,clobber=kw['clobber'],unlock=kw['unlock'])

def flat(name,**kw):
    """ Make Flat calibration files """
    load = kw['load']
    flatfile = load.filename('Flat',num=name,chips=True)
    flatdir = os.path.dirname(load.filename('Flat',num=name,chips=True))
    fmt = '{:s}Flat-{:s}-{:08d}.fits'
    outfiles = [os.path.join(flatdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    outfiles += [os.path.join(flatdir,load.prefix+'Flat-{:08d}.tab'.format(int(name)))]
    if np.sum([os.path.exists(f) for f in outfiles]==4 and clobber==False:
        print(' flat file: ',flatfile+'.tab',' already made')
        return
    flattab = kw['allcaldict']['flat'] 
    ind, = np.where(flattab['name']==str(name))
    if len(ind)==0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    ims = getnums(flattab[ind]['frames'])
    cmjd = int(load.cmjd(ims[0]))
    caldict = getcal(kw['calfile'],mjd)
    makecal(dark=darkid,unlock=kw['unlock'])
    cal.mkflat(ims,darkid=darkid,nrep=flattab[ind]['nrep'],dithered=flattab[ind]['dithered'],
               clobber=kw['clobber'],unlock=kw['unlock'])

def bpm(name,**kw):
    """ Make BPM calibration files """
    load = kw['load']
    bpmfile = load.filename('BPM',num=name,chips=True)
    bpmdir = os.path.dirname(load.filename('BPM',num=name,chips=True))
    fmt = '{:s}BPM-{:s}-{:08d}.fits'
    outfiles = [os.path.join(bpmdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' bpm file: ',bpmfile, ' already made')
        return
    bpmtab = kw['allcaldict']['bpm']
    ind, = np.where(bpmtab['name']==str(name))
    if len(ind) == 0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    makecal(dark=bpmtab[ind]['darkid'],unlock=kw['unlock'])
    makecal(flat=bpmtab[ind]['flatid'],unlock=kw['unlock'])
    cal.mkbpm(bpmtab[ind]['name'],darkid=bpmtab[ind]['darkid'],flatid=bpmtab[ind]['flatid'],
              clobber=kw['clobber'],unlock=kw['unlock'])

def sparse(name,**kw):
    """ Make Sparsepak PSF calibration product """
    load = kw['load']
    sparsefile = load.filename('Sparse',num=name,chips=True)
    psfdir = os.path.dirname(load.filename('Sparse',num=name,chips=True))
    if os.path.exists(sparsefile) and clobber==False:
        print(' sparse file: ',sparsefile,' already made')
        return
    sparsetab = kw['allcaldict']['sparse'] 
    ind, = np.where(sparsetab['name']==str(name))
    if len(ind) < 0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    ims = getnums(sparsetab[ind]['frames'])
    mjd = int(load.cmjd(ims[0))
    caldict = getcal(kw['calfile'],mjd)
    makecal(dark=darkid,unlock=kw['unlock'])
    makecal(flat=flatid,unlock=kw['unlock'])
    makecal(bpm=bpmid,unlock=kw['unlock'])
    darkims = getnums(sparsetab[ind]['darkframes'])
    maxread = getnums(sparsetab[ind]['maxread'])
    if len(maxread) != 3:
        print('sparse maxread does not have 3 elements! ')
        return
    cal.mkepsf(ims,darkid=darkid,flatid=flatid,darkims=darkims,dmax=sparsestr[i].dmax,
               maxread=maxread,clobber=kw['clobber'],filter=True,thresh=0.2,scat=2,unlock=kw['unlock'])
    # This creates apSparse and apEPSF files
    # Make empty apPSF files to indicate to makecal.pro that this
    #  PSF file was already made
    ssparse = string(sparse,format='(i08)')
    psffiles = psfdir+'/'+[dirs.prefix+'PSF-'+chips+'-'+ssparse+'.fits']  
    touchzero(psffiles)

def fiber(name,**kw):
    """ Make fiber calibration file """
    load = kw['load']
    psffile = load.filename('PSF',num=name,chips=True)
    psfdir = os.path.dirname(load.filename('PSF',num=name,chips=True))
    fmt = '{:s}{:s}-{:s}-{:08d}.fits'
    outfiles = [os.path.join(psfdir,fmt.format(load.prefix,'EPSF',ch,int(name))) for ch in chips]
    outfiles = [os.path.join(psfdir,fmt.format(load.prefix,'PSF',ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==6 and clobber==False:
        print(' psf file: ',psffile, ' already made')
        return
    cmjd = int(load.cmjd(name))
    caldict = getcal(kw['calfile'],mjd)
    cal.mkpsf(name,darkid=caldict['darkid'],flatid=caldict['flatid'],
              sparseid=caldict['sparseid'],unlock=kw['unlock'])

def psf(name,**kw):
    """ Make PSF calibration file """
    load = kw['load']
    #if keyword_set(psf) and not keyword_set(flux) and not keyword_set(wave)
    psffile = load.filename('PSF',num=name,chips=True)
    psfdir = os.path.dirname(load.filename('PSF',num=name,chips=True))
    fmt = '{:s}{:s}-{:s}-{:08d}.fits'
    outfiles = [os.path.join(psfdir,fmt.format(load.prefix,'EPSF',ch,int(name))) for ch in chips]
    outfiles += [os.path.join(psfdir,fmt.format(load.prefix,'PSF',ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==6 and clobber==False:
        print(' psf file: ',psffile, ' already made')
        return
    cmjd = int(load.cmjd(name))
    caldict = getcal(kw['calfile'],mjd)
    makecal(littrow=littrowid,unlock=kw['unlock'])
    cal.mkpsf(psf,bpmid=bpmid,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,
              littrowid=littrowid,clobber=kw['clobber'],unlock=kw['unlock'])

def modelpsf(name,**kw):
    """ Make Model PSF calibration file """
    #if keyword_set(modelpsf) and (not keyword_set(fpi) and not keyword_set(flux) and not keyword_set(wave)) then begin
    load = kw['load']
    psffile = load.filename('PSFModel',num=name,chips=True)
    psfdir = os.path.dirname(load.filename('PSFModel',num=name,chips=True))
    fmt = '{:s}PSFModel-{:s}-{:08d}.fits'
    allfiles = [os.path.join(psfdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfile])==3 and clobber==False:
        print(' modelpsf file: ',psffile, ' already made')
        return
    modelpsftab = kw['allcaldict']['modelpsf'] 
    ind, = np.where(modelpsftab['name']==str(name))
    if len(ind) < 0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    makecal(sparse=modelpsftab[ind]['sparse'],unlock=kw['unlock'])
    makecal(psf=modelpsftab[ind]['psf'],unlock=kw['unlock'])
    cal.mkmodelpsf(name,sparseid=modelpsftab[ind]['sparse'],psfid=modelpsftab[ind]['psf'],
                   clobber=kw['clobber'],unlock=kw['unlock'])

def fpi(name,**kw):
    """ Make FPI calibration file """
    load = kw['load']
    wavefpifile = load.filename('WaveFPI',num=name,chips=True)
    wavefpidir = os.path.dirname(load.filename('WaveFPI',num=name,chips=True))
    cmjd = load.cmjd(name)
    mjd = int(mjd)
    fmt = '{:s}WaveFPI-{:s}-{:s}-{:08d}.fits'
    outfiles = [os.path.join(wavedir,fmt.format(load.prefix,ch,cmjd,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print,' fpi file: ',file, ' already made'
        return
    caldict = getcal(kw['calfile'],mjd)
    # Use Model PSF by default
    if not keyword_set(psf) and not keyword_set(librarypsf):
        psfid = 0
        modelpsf = psfmodel
        makecal(modelpsf=modelpsf,unlock=kw['unlock'])
    # Use PSF file     
    else:
        # What PSF to use
        if keyword_set(psf):
            psfid = psf
        # Try to find a PSF from this day
        else:
            print,'Trying to automatically find a PSF calibration file'
            psfid = getpsfcal(fpi[0],psflibrary=librarypsf,unlock=kw['unlock'])
        makecal(psf=psfid,unlock=kw['unlock'])
        
    makecal(fiber=fiberid,unlock=kw['unlock'])
    makecal(dailywave=mjd,unlock=kw['unlock'],librarypsf=librarypsf,modelpsf=modelpsf)
    cal.mkfpi(fpi,name=name,darkid=darkid,flatid=flatid,psfid=psfid,
              fiberid=fiberid,clobber=kw['clobber'],unlock=kw['unlock'],psflibrary=librarypsf,modelpsf=modelpsf)

def littrow(name,**kw):
    """ Make Littrow calibration file """
    load = kw['load']
    litfile = load.filename('Littrow',num=name,chips=True)
    if os.path.exists(litfile) and clobber==False:
        print(' littrow file: ',litfile,' already made')
        return
    cmjd = load.cmjd(name)
    mjd = int(cmjd)
    caldict = getcal(kw['calfile'],mjd)
    makecal(flat=flatid,unlock=kw['unlock'])
    cal.mklittrow(littrow,cmjd=cmjd,darkid=darkid,flatid=flatid,sparseid=sparseid,
                  fiberid=fiberid,clobber=kw['clobber'],unlock=kw['unlock'])

def persist(name,**kw):
    """ Make Persistence calibration file """
    load = kw['load']
    perfile = load.filename('Persist',num=name,chips=True)
    perdir = os.path.dirname(load.filename('Persist',num=name,chips=True))
    fmt = '{:s}Persist-{:s}-{:08d}.fits'
    outfiles = [os.path.join(perdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' persist file: ',perfile, ' already made')
        return
    persisttab = kw['allcaldict']['persist'] 
    ind, = np.where(persisttab['name']==str(name))
    if len(ind) <= 0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    cmd = load.cmjd(name)
    mjd = int(cmjd)
    caldict = getcal(kw['calfile'],mjd)
    cal.mkpersist(persist,persisttab[ind]['darkid'],persisttab[ind]['flatid'],
                  thresh=persisttab[ind]['thresh'],cmjd=cmjd,darkid=darkid,flatid=flatid,
                  sparseid=sparseid,fiberid=fiberid,clobber=kw['clobber'],unlock=kw['unlock'])

def persistmodel(name,**kw):
    """ Make Persistence model calibration file """
    load = kw['load']
    perfile = load.filename('PersistModel',num=name,chips=True)
    perdir = os.path.dirname(load.filename('PersistModel',num=name,chips=True))
    fmt = '{:s}PersistModel-{:s}-{:08d}.fits'
    outfiles = [os.path.join(perdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' modelpersist file: ',file, ' already made')
        return
    persistmodeltab = kw['allcaldict']['persistmodel'] 
    ind, = np.where(persistmodeltab['name']==str(name))
    if len(ind)<0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    mjd = int(load.cmjd(name))
    caldict = getcal(kw['calfile'],mjd)
    cal.mkpersistmodel(name)

def flux(name,**kw):
    """ Make Flux calibration file """
    load = kw['load']
    fluxfile = load.filename('Flux',num=name,chips=True)
    fluxdir = os.path.dirname(load.filename('Flux',num=name,chips=True))
    fmt = '{:s}Flux-{:s}-{:08d}.fits'
    outfiles = [os.path.join(fluxdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' flux file: ',fluxfile, ' already made')
        return
    mjd = int(load.cmjd(name))
    caldict = getcal(kw['calfile'],mjd)
    # Use Model PSF by default
    if not keyword_set(psf) and not keyword_set(librarypsf) and keyword_set(psfmodel):
        psfid = 0
        modelpsf = psfmodel
        makecal(modelpsf=modelpsf,unlock=kw['unlock'])
    # Use PSF file     
    else:
        if keyword_set(psf):
            psfid = psf
        # Try to find a PSF from this day
        else:
            print('Trying to automatically find a PSF calibration file')
            psfid = getpsfcal(flux[0],psflibrary=librarypsf,unlock=kw['unlock'])
        makecal(psf=psfid,unlock=kw['unlock'])
        
    makecal(littrow=littrowid,unlock=kw['unlock'])
    cal.mkflux(flux,darkid=darkid,flatid=flatid,psfid=psfid,modelpsf=modelpsf,
               littrowid=littrowid,waveid=waveid,clobber=kw['clobber'],unlock=kw['unlock'])

def response(name,**kw):
    """ Make Response calibration file """
    load = kw['load']
    resfile = load.filename('Response',num=name,chips=True)
    resdir = os.path.dirname(load.filename('Response',num=name,chips=True))
    fmt = '{:s}Response-{:s}-{:08d}.fits'
    outfiles = [os.path.join(resdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' response file: ',resfile, ' already made')
        return
    responsetab = kw['allcaldict']['response'] 
    ind, = np.where(responsetab['name']==str(name))
    if len(ind) == 0:
        print('No matching calibration line for',name)
    else:
        if nres>1:
            i=i[0]
    mjd = int(load.cmjd(name))
    caldict = getcal(kw['calfile'],mjd)
    makecal(psf=responsetab[ind]['psf'],unlock=kw['unlock'])
    makecal(wave=waveid,unlock=kw['unlock'])
    makecal(fiber=fiberid,unlock=kw['unlock'])
    makecal(littrow=littrowid,unlock=kw['unlock'])
    cal.mkflux(response,darkid=darkid,flatid=flatid,psfid=responsetab[ind]['psf'],
               littrowid=littrowid,waveid=waveid,temp=responsetab[ind]['temp'],
               clobber=kw['clobber'],unlock=kw['unlock'])

def wave(name,**kw):
    """ Make Wavelength calibration file """
    load = kw['load']
    wavefile = load.filename('Wave',num=name,chip='c')
    wavedir = os.path.dirname(load.filename('Wave',num=name,chip='c'))
    fmt = '{:s}Wave-{:s}-{:08d}.fits'
    outfiles = [os.path.join(wavedir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' wave file: ',wavefile, ' already made')
        return
    wavetab = kw['allcaldict']['wave'] 
    ind, = np.where(wavetab['name']==str(name))
    if len(ind) > 0:
        ims = getnums(wavetab[ind[0]]['frames'])
        name = wavetab[ind[0]]['name']
        psfid = wavetab[ind[0]]['psfid']
    # Use the input filename
    else:
        ims = wave
        name = ims[0]
        mjd = int(load.cmjd(name))
        caldict = getcal(kw['calfile'],mjd)
        # Use Model PSF by default
        if not keyword_set(psf) and not keyword_set(librarypsf):
            psfid = 0
            modelpsf = psfmodel
            makecal(modelpsf=modelpsf,unlock=kw['unlock'])
        # Use PSF file     
        else:
            if keyword_set(psf):
                psfid = psf
            # Try to find a PSF from this day
            else:
                print('Trying to automatically find a PSF calibration file')
                psfid = getpsfcal(ims[0],psflibrary=librarypsf,unlock=kw['unlock'])
            makecal(psf=psfid,unlock=kw['unlock'])
    mjd = int(load.cmjd(ims[0]))
    caldict = getcal(kw['calfile'],mjd)
    makecal(bpm=bpmid,unlock=kw['unlock'])
    makecal(fiber=fiberid,unlock=kw['unlock'])
    cal.mkwave(ims,name=name,darkid=darkid,flatid=flatid,psfid=psfid,modelpsf=modelpsf,
               fiberid=fiberid,clobber=kw['clobber'],nofit=nofit,unlock=kw['unlock'])
            
def multiwave(name,**kw):
    load = kw['load']
    """ Make multi-night wavelength calibration file """
    wavefile = load.filename('Wave',num=name,chips=True)
    wavedir = os.path.dirname(load.filename('Wave',num=name,chips=True))
    fmt = '{:s}Wave-{:s}-{:08d}.fits'
    outfiles = [os.path.join(wavedir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    outfiles += [os.path.join(wavedir,load.prefix+'Wave-{:08d}.py.dat'.format(int(name)))]
    if np.sum([os.path.exists(f) for f in outfiles])==4 and clobber==False:
        print(' multiwave file: ',wavefile+'.dat',' already made')
        return
    multiwavetab = kw['allcaldict']['multiwave'] 
    ind, = np.where(multiwavetab['name']==str(name))
    if len(ind)==0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    ims = getnums(multiwavetab[ind[0]]['frames'])
    cal.mkmultiwave(ims,name=multiwavetab[ind[0]]['name'],clobber=kw['clobber'],file=file,
                    unlock=kw['unlock'],psflibrary=librarypsf)

def dailywave(name,**kw):
    """ Make daily wavelength calibration file """
    load = kw['load']
    wavefile = load.filename('Wave',num=name,chips=True)
    wavedir = os.path.dirname(load.filename('Wave',num=name,chips=True))
    fmt = '{:s}Wave-{:s}-{:08d}.fits'
    outfiles = [os.path.join(wavedir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' dailywave file: ',wavefile,' already made')
        return
    mjd = int(name)
    caldict = getcal(kw['calfile'],mjd)
    #if keyword_set(librarypsf) then apgundef,modelpsf
    makecal(bpm=bpmid,unlock=kw['unlock'])
    makecal(fiber=fiberid,unlock=kw['unlock'])
    cal.mkdailywave(name,darkid=darkid,flatid=flatid,psfid=psfid,
                    fiberid=fiberid,clobber=kw['clobber'],nofit=nofit,unlock=kw['unlock'],
                    psflibrary=librarypsf,modelpsf=modelpsf)

def telluric(name,**kw):
    """ Make daily telluric calibration file """
    load = kw['load']
    telfile = load.filename('Telluric',num=name,chips=True)
    teldir = os.path.dirname(load.filename('Telluric',num=name,chips=True))
    fmt = '{:s}Telluric-{:s}-{:08d}.fits'
    outfiles = [os.path.join(teldir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    outfiles += [os.path.join(teldir,load.prefix+'Telluric-{:08d}.dat'.format(int(name)))]
    if np.sum([os.path.exists(f) for f in outfiles])==4 and clobber==False:
        print(' telluric file: ',telfile,' already made')
        return
    waveid = int(telluric.split('-')[0])
    lsfid = int(telluric.split('-')[1])
    if waveid < 1e7:
       makecal(dailywave=waveid,unlock=kw['unlock'])
    else:
       makecal(wave=waveid,unlock=kw['unlock'])
    makecal(lsf=lsfid,unlock=kw['unlock'])
    cal.mktelluric(name,clobber=kw['clobber'],unlock=kw['unlock'])

def lsf(name,**kw):
    """ Make LSF calibration file """
    load = kw['load']
    lsffile = load.filename('LSF',num=name,chips=True)
    lsfdir = os.path.dirname(load.filename('LSF',num=name,chips=True))
    fmt = '{:s}LSF-{:s}-{:08d}.fits'
    outfiles = [os.path.join(lsfdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    outfiles += [os.path.join(lsfdir,load.prefix+'LSF-{:08d}.sav'.format(int(name)))]
    if np.sum([os.path.exists(f) for f in outfiles])==4 and clobber==False:
        print(' lsf file: ',file+'.sav',' already made')
        return
    lsftab = kw['allcaldict']['lsf'] 
    ind, = np.where(lsftab['name']==str(name))
    if len(ind) <= 0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    ims = getnums(lsftab[ind[0]]['frames'])
    mjd = int(load.cmjd(ims[0]))
    caldict = getcal(kw['calfile'],mjd)
    makecal(multiwave=waveid,unlock=kw['unlock'])
    cal.mklsf(ims,name,darkid=darkid,flatid=flatid,psfid=lsftab[ind[0]]['psfid'],fiberid=fiberid,
              full=full,newwave=newwave,clobber=kw['clobber'],pl=pl,unlock=kw['unlock'])
