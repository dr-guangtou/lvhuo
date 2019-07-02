import os
import copy
import scipy

import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column
import astropy.units as u
from astropy.coordinates import SkyCoord
from .display import display_single, SEG_CMAP


def psf_construct(img, x, y, halosize, norm=None, magnify=1, pixel_scale=0.168, 
                  sub_indiv_bkg=False, show_figure=False, verbose=False):
    
    from pyraf import iraf
    from iraf import stsdas
    from iraf import analysis
    from iraf import isophote
    from iraf import bmodel
    from sep import Background

    from .image import save_to_fits, extract_obj, seg_remove_cen_obj

    if not os.path.isdir('./temp/'):
        os.mkdir('./temp/')
    # Delete `fits` file first
    iraf.imdel('./temp/_halo*.fits')

    x_int = x.astype(np.int)
    y_int = y.astype(np.int)
    dx = -1.0 * (x - x_int)
    dy = -1.0 * (y - y_int)
    
    halosize = int(halosize)

    # Note that stars near the boundary will have problems when making cutout. So we pad the image here.
    padsize = 40
    ny, nx = img.shape
    im_padded = np.zeros((ny + 2 * padsize, nx + 2 * padsize))
    # Making the left edge empty
    im_padded[padsize: ny + padsize, padsize: nx + padsize] = img
            
    halo_i = (im_padded[y_int + padsize - halosize: y_int + padsize + halosize + 1, 
                        x_int + padsize - halosize: x_int + padsize + halosize + 1])
    if verbose:                    
        print('### Cuting out the star and masking out contaminations ###')
    # Build mask to move out contaminations
    psf_raw = halo_i.byteswap().newbyteorder()
    from astropy.convolution import convolve, Box2DKernel
    psf_blur = convolve(abs(psf_raw), Box2DKernel(1.5))
    psf_objects, psf_segmap = extract_obj(abs(psf_blur), b=5, f=4, sigma=5.5, minarea=2, pixel_scale=pixel_scale,
                                          deblend_nthresh=32, deblend_cont=0.0001, 
                                          sky_subtract=False, show_fig=show_figure, verbose=verbose)
    
    # remove central object
    psf_segmap = seg_remove_cen_obj(psf_segmap) 
    psf_mask = (psf_segmap != 0).astype(float)
    save_to_fits(psf_mask, './temp/_halo_mask.fits')
    # Here I subtract local sky, so that the final image will not be too negative.
    if sub_indiv_bkg is True:
        # Evaluate local sky backgroud within `halo_i`
        # Actually this should be estimated in larger cutuouts.
        # So make another cutout (larger)!
        psf_raw = psf_raw.byteswap().newbyteorder()
        bk = Background(psf_raw, psf_segmap != 0)
        glbbck = bk.globalback
        if verbose:
            print('# Global background: ', glbbck)
        save_to_fits(halo_i - glbbck, './temp/_halo_img.fits') #
    else:
        save_to_fits(halo_i, './temp/_halo_img.fits') #

    # Shift halo_i to integer grid
    iraf.imshift('./temp/_halo_img.fits', './temp/_halo_img_shift.fits',
                 dx, dy, interp_type='poly3', 
                 boundary='nearest')
    iraf.imshift('./temp/_halo_mask.fits', './temp/_halo_mask_shift.fits',
                 dx, dy, interp_type='poly3', 
                 boundary='nearest')
    
    if not isinstance(magnify, int):
        raise TypeError('# "magnify" must be positive integer less than 10!')
    elif magnify != 1 and magnify > 10:
        raise TypeError('# "magnify" must be positive integer less than 10!')
    elif magnify == 1:
        iraf.imcopy('./temp/_halo_img_shift.fits', './temp/_halo_i_shift.fits', verbose=verbose)
        iraf.imcopy('./temp/_halo_mask_shift.fits', './temp/_halo_m_shift.fits', verbose=verbose)
    else:
        iraf.magnify('./temp/_halo_img_shift.fits', './temp/_halo_i_shift.fits', 
             magnify, magnify, interpo='poly3', 
             bound='refl', const=0.)
        iraf.magnify('./temp/_halo_mask_shift.fits', './temp/_halo_m_shift.fits', 
             magnify, magnify, interpo='poly3', 
             bound='refl', const=0.)

    return psf_raw, psf_mask #, x_int, y_int, dx, dy
