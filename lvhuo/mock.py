import os
import copy

import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column, hstack
import astropy.units as u
from astropy.coordinates import SkyCoord

def single_sersic_galaxy(gal_params, size=[60, 60], psf_rh=3.0, psf_beta=None, psf_custom=None, 
                         add_noise=None, band='r', pixel_scale=0.168, pixel_unit=False):
    """
    Generate single Sersic galaxy
    """
    import galsim
    # Define sersic galaxy
    gal = galsim.Sersic(gal_params['sersic_n'], half_light_radius=gal_params['gal_rh'],
                        flux=gal_params['gal_flux'])
    # Shear the galaxy by some value.
    gal_shape = galsim.Shear(q=gal_params['gal_q'], 
                             beta=gal_params['gal_beta'] * galsim.degrees)
    gal = gal.shear(gal_shape)
    # Define the PSF profile
    if psf_custom is not None:
        psf = galsim.InterpolatedImage(galsim.Image(psf_custom, dtype=float, scale=0.168))
    elif psf_rh < 0:
        raise ValueError('PSF half-light radius (`psf_rh`) must be positive!')
    elif psf_beta is not None:
        psf = ggalsim.Moffat(beta=psf_beta, flux=1., half_light_radius=psf_rh)
    else:
        psf = galsim.Gaussian(sigma=psf_rh, flux=1.)
    # Convolve galaxy with PSF
    final = galsim.Convolve([gal, psf])
    # Draw the image with a particular pixel scale.
    if pixel_unit:
        image = final.drawImage(scale=pixel_scale, nx=size[1], ny=size[0])
    else:
        image = final.drawImage(scale=pixel_scale, nx=size[1] // pixel_scale, ny=size[0] // pixel_scale)
    # Add noise if want
    if add_noise is not None:
        #assert isinstance(add_noise, float) or isinstance(add_noise, int), '`add_noise` must be float!'
        noise = galsim.GaussianNoise(sigma=add_noise)
        image.addNoise(noise)
    return image