# encoding: utf-8
#
# @Author: Jon Holtzman
# @Date: July 2018
# @Filename: norm.py
# @License: BSD 3-Clause
# @Copyright: Jon Holtzman

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import copy
import pdb
from astropy.io import fits
from astropy.io import ascii
#from apogee.aspcap import aspcap
#from apogee.aspcap import ferre
from scipy.ndimage.filters import median_filter
from scipy import interpolate
from holtztools import plots


logw0=4.179
dlogw=6.e-6
nw_apStar=8575
def apStarWave() :
    """ Returns apStar wavelengths
    """
    return 10.**(logw0+np.arange(nw_apStar)*dlogw)

logw0_chip=np.array([4.180476,4.200510,4.217064])
nw_chip=np.array([3028,2495,1991])
def gridWave() :
    """ Returns aspcap grid wavelengths
    """
    return [10.**(logw0_chip[0]+np.arange(nw_chip[0])*dlogw),
            10.**(logw0_chip[1]+np.arange(nw_chip[1])*dlogw),
            10.**(logw0_chip[2]+np.arange(nw_chip[2])*dlogw)]

def gridPix(apStar=True) :
    """ Returns chip pixel ranges in apStar or aspcap grid
    """
    if apStar :
        w=np.log10(apStarWave())
        s1 = np.where(np.isclose(w,logw0_chip[0],rtol=0.))[0][0]
        s2 = np.where(np.isclose(w,logw0_chip[1],rtol=0.))[0][0]
        s3 = np.where(np.isclose(w,logw0_chip[2],rtol=0.))[0][0]
        e1 = np.where(np.isclose(w,logw0_chip[0]+nw_chip[0]*dlogw,rtol=0.))[0][0]
        e2 = np.where(np.isclose(w,logw0_chip[1]+nw_chip[1]*dlogw,rtol=0.))[0][0]
        e3 = np.where(np.isclose(w,logw0_chip[2]+nw_chip[2]*dlogw,rtol=0.))[0][0]
        return [[s1,e1],[s2,e2],[s3,e3]]
    else :
        return [[0,nw_chip[0]],[nw_chip[0],nw_chip[0]+nw_chip[1]],[nw_chip[0]+nw_chip[1],nw_chip[0]+nw_chip[1]+nw_chip[2]]]


def cont(spec,specerr,chips=False,order=4,poly=True,apstar=True,medfilt=0) :
    """ Returns continuum normalized spectrum
    """
    x = np.arange(0,len(spec))
   
    if chips :
        cont = np.full_like(spec,np.nan)
        pranges = gridPix(apStar=apstar)
        for prange in pranges :
            s = spec[prange[0]:prange[1]]
            serr = specerr[prange[0]:prange[1]]
            xx = x[prange[0]:prange[1]]
            if poly :
                cont[prange[0]:prange[1]] = polyfit(xx,s,serr,order)
            else :
                cont[prange[0]:prange[1]] = median_filter(s,[medfilt],mode='reflect')
    else :
        if poly :
            cont = polyfit(x,spec,specerr,order)
        else :
            cont = median_filter(spec,[medfilt],mode='reflect')

    return cont

def polyfit(x,y,yerr,order) :
    """ continuum via polynomial fit
    """
    gd = np.where(np.isfinite(y))[0]
    # note unconventional definition in numpy.polyfit for weights!
    p = np.poly1d(np.polyfit(x[gd],y[gd],order,w=1./yerr[gd]))
    return p(x)

