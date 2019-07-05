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
from .image import img_cutout

class Stack():
    '''
    Class for `Stack` object.
    '''
    def __init__(self, img, mask=None, header=None, data_release='s18a'):
        '''Initialize stack object'''
        self.header = header
        self.wcs = wcs.WCS(header)
        self.size = img.shape # in ndarray format
        self.data_release = data_release
        self._image = img
        self._mask = mask
        # Sky position
        ny, nx = img.shape
        self.ny = ny
        self.nx = nx
        self.ra_cen, self.dec_cen = list(map(float, self.wcs.wcs_pix2world(ny // 2, nx // 2, 0)))
        # This follows lower-left, lower-right, upper-right, upper-left.
        self.ra_bounds, self.dec_bounds = self.wcs.wcs_pix2world([0, img.shape[1], img.shape[1], 0], 
                                            [0, 0, img.shape[0], img.shape[0]], 0)
        self.sky_bounds = np.append(self.ra_bounds[2:], self.dec_bounds[1:3])
        
    @property
    def image(self):
        return self._image
    
    @image.setter
    def image(self, img_array):
        self._image = img_array

    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, mask_array):
        self._mask = mask_array
    
    @property
    def hscmask(self):
        return self._hscmask
    
    @hscmask.setter
    def hscmask(self, mask_array):
        self._hscmask = mask_array

    @property
    def variance(self):
        return self._variance
    
    @variance.setter
    def variance(self, variance_array):
        self._variance = variance_array

    # Save 2-D numpy array to `fits`
    def save_to_fits(self, fits_file_name, overwrite=True):
        """Save numpy 2-D arrays to `fits` file. (from `kungpao`)
        Parameters:
            fits_file_name (str): File name of `fits` file
            overwrite (bool): Default is True

        Returns:
            None
        """
        if self.wcs is not None:
            wcs_header = self.wcs.to_header()
            img_hdu = fits.PrimaryHDU(self.image, header=wcs_header)
        else:
            img_hdu = fits.PrimaryHDU(self.image)

        if self.header is not None:
            img_hdu.header = self.header

        if os.path.islink(fits_file_name):
            os.unlink(fits_file_name)

        img_hdu.writeto(fits_file_name, output_verify='warn', overwrite=overwrite)

    # Rotate image/mask
    def rotate_image(self, angle, method='lanczos', order=5, reshape=False, cval=0.0):
        '''Rotate the image of Stack object.

        Parameters:
            angle (float): rotation angle in degress, counterclockwise.
            method (str): interpolation method. Use 'lanczos', 'spline', 'cubic', 
                'bicubic', 'nearest' or 'bilinear'.
            order (int): the order of spline interpolation (within 0-5) or Lanczos interpolation (>0).
            reshape (bool): if True, the output shape is adapted so that the rorated image 
                is contained completely in the output array.
            cval (scalar): value to fill the edges. Default is NaN.
        
        Returns:
            rotate_image: ndarray.
        '''
        angle = angle % 360

        if method == 'lanczos':
            try:
                from galsim import degrees, Angle
                from galsim.interpolant import Lanczos
                from galsim import Image, InterpolatedImage
                from galsim.fitswcs import AstropyWCS
            except:
                raise ImportError('# Import `galsim` failed! Please check if `galsim` is installed!')
            # Begin rotation
            assert (order > 0) and isinstance(order, int), 'order of ' + method + ' must be positive interger.'
            galimg = InterpolatedImage(Image(self.image, dtype=float), 
                                       scale=0.168, x_interpolant=Lanczos(order))
            galimg = galimg.rotate(Angle(angle, unit=degrees))
            ny, nx = self.image.shape
            result = galimg.drawImage(scale=0.168, nx=nx, ny=ny)#, wcs=AstropyWCS(self.wcs))
            self._image = result.array
            return result.array

        elif method == 'spline':
            from scipy.ndimage.interpolation import rotate as rt
            assert 0 < order <= 5 and isinstance(order, int), 'order of ' + method + ' must be within 0-5.'
            result = rt(self.image, angle, order=order, mode='constant', 
                        cval=cval, reshape=reshape)
            self._image = result
            return result
        elif method in ['bicubic', 'nearest','cubic','bilinear']:
            try:
                from scipy.misc import imrotate
            except:
                raise ImportError('# Import `scipy.misc.imrotate` failed! This function may no longer be included in scipy!')
            result = imrotate(self.image, angle, interp=method)
            self._image = result
            return result
        else:
            raise ValueError("# Not supported interpolation method. Use 'lanczos', 'spline', 'cubic', \
                             'bicubic', 'nearest', 'bilinear'.")

    def rotate_mask(self, angle, method='lanczos', order=5, reshape=False, cval=0.0):
        '''Rotate the mask of Stack object.

        Parameters:
            angle (float): rotation angle in degress, counterclockwise.
            method (str): interpolation method. Use 'lanczos', 'spline', 'cubic', 
                'bicubic', 'nearest' or 'bilinear'.
            order (int): the order of spline interpolation (within 0-5) or Lanczos interpolation (>0).
            reshape (bool): if True, the output shape is adapted so that the rorated image 
                is contained completely in the output array.
            cval (scalar): value to fill the edges. Default is NaN.
        
        Returns:
            rotate_image: ndarray.
        '''
        angle = angle % 360

        if not hasattr(self, 'mask'):
            raise AttributeError("This `Stack` object doesn't have `mask`!")
        else: 
            if method == 'lanczos':
                try:
                    from galsim import degrees, Angle
                    from galsim.interpolant import Lanczos
                    from galsim import Image, InterpolatedImage
                    from galsim.fitswcs import AstropyWCS
                except:
                    raise ImportError('# Import `galsim` failed! Please check if `galsim` is installed!')
                # Begin rotation
                assert (order > 0) and isinstance(order, int), 'order of ' + method + ' must be positive interger.'
                galimg = InterpolatedImage(Image(self.mask, dtype=float), 
                                        scale=0.168, x_interpolant=Lanczos(order))
                galimg = galimg.rotate(Angle(angle, unit=degrees))
                ny, nx = self.image.shape
                result = galimg.drawImage(scale=0.168, nx=nx, ny=ny) #, wcs=AstropyWCS(self.wcs))
                self._mask = (result.array > 0.5).astype(float)
                return result.array

            elif method == 'spline':
                from scipy.ndimage.interpolation import rotate as rt
                assert 0 < order <= 5 and isinstance(order, int), 'order of ' + method + ' must be within 0-5.'
                result = rt(self.mask, -angle, order=order, mode='constant', 
                            cval=cval, reshape=reshape)
                self._mask = (result > 0.5).astype(float)
                return result
            elif method in ['bicubic', 'nearest','cubic','bilinear']:
                try:
                    from scipy.misc import imrotate
                except:
                    raise ImportError('# Import `scipy.misc.imrotate` failed! This function may no longer be included in scipy!')
                result = imrotate(self.mask, -angle, interp=method)
                self._mask = (result > 0.5).astype(float)
                return result
            else:
                raise ValueError("# Not supported interpolation method. Use 'lanczos', 'spline', 'cubic', \
                                'bicubic', 'nearest', 'bilinear'.")

    def rotate_Stack(self, angle, method='lanczos', order=5, reshape=False, cval=0.0):
        '''Rotate the Stack object.

        Parameters:
            angle (float): rotation angle in degress, counterclockwise.
            order (int): the order of spline interpolation, can be in the range 0-5.
            reshape (bool): if True, the output shape is adapted so that the rorated image 
                is contained completely in the output array.
            cval (scalar): value to fill the edges. Default is NaN.
        
        Returns:
        '''
        self.rotate_image(angle, method=method, order=order, reshape=reshape, cval=cval)
        if hasattr(self, 'mask'):
            self.rotate_mask(angle, method=method, order=order, reshape=reshape, cval=cval)

    # Shift image/mask
    def shift_image(self, dx, dy, method='lanczos', order=5, cval=0.0):
        '''Shift the image of Stack object.

        Parameters:
            dx, dy (float): shift distance (in pixel) along x (horizontal) and y (vertical). 
                Note that elements in one row has the same y but different x. 
                Example: dx = 2 is to shift the image "RIGHT", dy = 3 is to shift the image "UP".
            method (str): interpolation method. Use 'lanczos' or 'spline'.
            order (int): the order of spline interpolation (within 0-5) or Lanczos interpolation (>0).
            cval (scalar): value to fill the edges. Default is NaN.

        Returns:
            shift_image: ndarray.
        '''
        ny, nx = self.image.shape
        if abs(dx) > nx or abs(ny) > ny:
            raise ValueError('# Shift distance is beyond the image size.')
        if method == 'lanczos':
            try: # try to import galsim
                from galsim import degrees, Angle
                from galsim.interpolant import Lanczos
                from galsim import Image, InterpolatedImage
                from galsim.fitswcs import AstropyWCS
            except:
                raise ImportError('# Import `galsim` failed! Please check if `galsim` is installed!')
            # Begin rotation
            assert (order > 0) and isinstance(order, int), 'order of ' + method + ' must be positive interger.'
            galimg = InterpolatedImage(Image(self.image, dtype=float), 
                                    scale=0.168, x_interpolant=Lanczos(order))
            galimg = galimg.shift(dx=dx * 0.168, dy=dy * 0.168)
            result = galimg.drawImage(scale=0.168, nx=nx, ny=ny)#, wcs=AstropyWCS(self.wcs))
            self._image = result.array
            return result.array
        elif method == 'spline':
            from scipy.ndimage.interpolation import shift
            assert 0 < order <= 5 and isinstance(order, int), 'order of ' + method + ' must be within 0-5.'
            result = shift(self.image, [dy, dx], order=order, mode='constant', cval=cval)
            self._image = result
            return result
        else:
            raise ValueError("# Not supported interpolation method. Use 'lanczos' or 'spline'.")

    def shift_mask(self, dx, dy, method='lanczos', order=5, cval=0.0):
        '''Shift the mask of Stack object.

        Parameters:
            dx, dy (float): shift distance (in pixel) along x (horizontal) and y (vertical). 
                Note that elements in one row has the same y but different x. 
                Example: dx = 2 is to shift the image "RIGHT", dy = 3 is to shift the image "UP".
            method (str): interpolation method. Use 'lanczos' or 'spline'.
            order (int): the order of spline interpolation (within 0-5) or Lanczos interpolation (>0).
            cval (scalar): value to fill the edges. Default is NaN.

        Returns:
            shift_mask: ndarray.
        '''
        if not hasattr(self, 'mask'):
            raise AttributeError("This `Stack` object doesn't have `mask`!")
        ny, nx = self.image.shape
        if abs(dx) > nx or abs(ny) > ny:
            raise ValueError('# Shift distance is beyond the image size.')
            
        if method == 'lanczos':
            try: # try to import galsim
                from galsim import degrees, Angle
                from galsim.interpolant import Lanczos
                from galsim import Image, InterpolatedImage
                from galsim.fitswcs import AstropyWCS
            except:
                raise ImportError('# Import `galsim` failed! Please check if `galsim` is installed!')
            # Begin rotation
            assert (order > 0) and isinstance(order, int), 'order of ' + method + ' must be positive interger.'
            galimg = InterpolatedImage(Image(self.mask, dtype=float), 
                                    scale=0.168, x_interpolant=Lanczos(order))
            galimg = galimg.shift(dx=dx * 0.168, dy=dy * 0.168)
            result = galimg.drawImage(scale=0.168, nx=nx, ny=ny)#, wcs=AstropyWCS(self.wcs))
            self._mask = (result.array > 0.5).astype(float)
            return result.array
        elif method == 'spline':
            from scipy.ndimage.interpolation import shift
            assert 0 < order <= 5 and isinstance(order, int), 'order of ' + method + ' must be within 0-5.'
            result = shift(self.mask, [dy, dx], order=order, mode='constant', cval=cval)
            self._mask = (result > 0.5).astype(float)
            return result
        else:
            raise ValueError("# Not supported interpolation method. Use 'lanczos' or 'spline'.")

    def shift_Stack(self, dx, dy, method='lanczos', order=5, cval=0.0):
        '''Shift the Stack object.

        Parameters:
            dx, dy (float): shift distance (in pixel) along x (horizontal) and y (vertical). 
                Note that elements in one row has the same y but different x. 
                Example: dx = 2 is to shift the image "RIGHT", dy = 3 is to shift the image "UP".
            method (str): interpolation method. Use 'lanczos' or 'spline'.
            order (int): the order of spline interpolation (within 0-5) or Lanczos interpolation (>0).
            cval (scalar): value to fill the edges. Default is NaN.
        
        Returns:
        '''
        self.shift_image(dx, dy, method=method, order=order, cval=cval)
        if hasattr(self, 'mask'):
            self.shift_mask(dx, dy, method=method, order=order, cval=cval)
    
    # Magnify image/mask
    

    # Display image/mask
    def display_image(self):
        display_single(self.image)
    def display_mask(self):
        display_single(self.mask, scale='linear', cmap=SEG_CMAP)
    def display_Stack(self):
        if hasattr(self, 'mask'):
            display_single(self.image * (~self.mask.astype(bool)))
        else:
            self.display_image()
       

class StackSky(Stack):
    def __init__(self, img, header, skyobj, mask=None, aper_name='aper57'):
        Stack.__init__(self, img, mask, header=header)
        self.name = 'sky'

        from unagi.sky import SkyObjs, AperPhot, S18A_APER, S18A_APER_ID
        cutout, cen_pos = img_cutout(self.image, self.wcs, skyobj['ra'], skyobj['dec'], 
                                     size=2 * S18A_APER[aper_name].r_arcsec, save=False)
        self._image = cutout.data
        self.size = self.image.shape
        self.cen_xy = cen_pos[0]
        self.dx = cen_pos[1]
        self.dy = cen_pos[2]

        if hasattr(self, 'mask'):
            cutout, _ = img_cutout(self.mask, self.wcs, skyobj['ra'], skyobj['dec'], 
                                   size=2 * S18A_APER[aper_name].r_arcsec, save=False)
            self._mask = cutout.data

    def centralize(self):
        self.shift_Stack(self.dx, self.dy, order=0, cval=0.0)
    
    def get_masked_image(self):
        if not hasattr(self, 'mask'):
            raise Warning("This `StackSky` object doesn't have a `mask`!")
            return self.image
        else:
            return self.image * (~self.mask.astype(bool))
        
    # Display image/mask
    def display_image(self):
        display_single(self.image, scale_bar_length=1)
    def display_mask(self):
        display_single(self.mask, scale='linear', cmap=SEG_CMAP, scale_bar_length=1)
    def display_Stack(self):
        if hasattr(self, 'mask'):
            display_single(self.image * (~self.mask.astype(bool)), scale_bar_length=1)
        else:
            self.display_image()
    

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

