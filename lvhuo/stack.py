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


class Stack(object):
    '''
    Class for `Stack` object.
    '''
    def __init__(self, img, mask=None, header=None, data_release='s18a'):
        '''Initialize stack object'''
        self.header = header
        try:
            self.pixel_scale = abs(header['CD1_1'] * 3600)
        except:
            self.pixel_scale = abs(header['PC1_1'] * 3600)
        self.wcs = wcs.WCS(header)
        self.shape = img.shape # in ndarray format
        self.data_release = data_release
        self._image = img
        if mask is not None:
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
        self.scale_bar_length = 5 # initial length for scale bar when displaying

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
    def save_to_fits(self, fits_file_name, data='image', overwrite=True):
        """Save numpy 2-D arrays to `fits` file. (from `kungpao`)
        Parameters:
            data (str): can be 'image' or 'mask'
            fits_file_name (str): File name of `fits` file
            overwrite (bool): Default is True

        Returns:
            None
        """
        if data == 'image':
            data_use = self.image
        elif data == 'mask':
            data_use = self.mask
        else:
            raise ValueError('Data can only be "image" or "mask".')
        img_hdu = fits.PrimaryHDU(data_use)

        if self.header is not None:
            img_hdu.header = self.header
            if self.wcs is not None:
                wcs_header = self.wcs.to_header()
                import fnmatch
                for i in self.header:
                    if i in wcs_header:
                        self.header[i] = wcs_header[i]
                    if fnmatch.fnmatch(i, 'CD?_?'):
                        self.header[i] = wcs_header['PC' + i.lstrip('CD')]
                img_hdu.header = self.header
        elif self.wcs is not None:
            wcs_header = self.wcs.to_header()
            img_hdu.header = wcs_header
        
        else:
            img_hdu = fits.PrimaryHDU(data_use)

        if os.path.islink(fits_file_name):
            os.unlink(fits_file_name)

        img_hdu.writeto(fits_file_name, overwrite=overwrite)
        return img_hdu

    def _rotate_wcs(self, img, w, theta):
        """Rotate the WCS of a given image. 
        See: https://github.com/astropy/astropy/issues/5175, https://arxiv.org/pdf/astro-ph/0207407.pdf.
        Parameters:
            img (2-D array): input image.
            w (astropy.wcs.WCS object): wcs of the input image.
            theta (float): rotation angle in degrees, counterclockwise.
        
        Return:
            wcs_temp: rotated WCS.
        """
        w_temp = copy.deepcopy(w)
        _theta = np.deg2rad(theta)
        # Rotation matrix
        _mrot = np.zeros(shape=(2, 2), dtype=np.double)
        _mrot[0] = (np.cos(_theta), np.sin(_theta))
        _mrot[1] = (-np.sin(_theta), np.cos(_theta))
        # New transforming matrix
        new_cd = np.dot(w_temp.wcs.cd, _mrot)
        w_temp.wcs.cd = new_cd
        # Rotate reference pixel position
        ref_pix = w_temp.wcs.crpix.reshape(-1, 1)
        ref_pix -= np.array([[img.shape[1]//2], [img.shape[0]//2]]) 
        # change the reference pixel relative to image center
        ref_pix = np.dot(_mrot.T, ref_pix)
        ref_pix += np.array([[img.shape[1]//2], [img.shape[0]//2]])
        w_temp.wcs.crpix = ref_pix.reshape(1, -1)[0]
        return w_temp

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
                                       scale=self.pixel_scale, x_interpolant=Lanczos(order))
            galimg = galimg.rotate(Angle(angle, unit=degrees))
            ny, nx = self.image.shape
            result = galimg.drawImage(scale=self.pixel_scale, nx=nx, ny=ny)#, wcs=AstropyWCS(self.wcs))
            self.wcs = self._rotate_wcs(self.image, self.wcs, angle)
            self._image = result.array
            self._wcs_header_merge()
            return result.array

        elif method == 'spline':
            from scipy.ndimage.interpolation import rotate as rt
            assert 0 < order <= 5 and isinstance(order, int), 'order of ' + method + ' must be within 0-5.'
            result = rt(self.image, -angle, order=order, mode='constant', 
                        cval=cval, reshape=reshape)
            
            self.wcs = self._rotate_wcs(self.image, self.wcs, angle)
            self._image = result
            self._wcs_header_merge()
            return result
        elif method in ['bicubic', 'nearest','cubic','bilinear']:
            raise Warning("Cautious! Don't use ['bicubic', 'nearest', 'cubic', 'bilinear'] methods! They don't conserve the total flux!")
            try:
                from scipy.misc import imrotate
            except:
                raise ImportError('# Import `scipy.misc.imrotate` failed! This function may no longer be included in scipy!')
            result = imrotate(self.image, angle, interp=method)
            self.wcs = self._rotate_wcs(self.image, self.wcs, angle)
            self._image = result.astype(bool)
            self._wcs_header_merge()
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
                                        scale=self.pixel_scale, x_interpolant=Lanczos(order))
                galimg = galimg.rotate(Angle(angle, unit=degrees))
                ny, nx = self.image.shape
                result = galimg.drawImage(scale=self.pixel_scale, nx=nx, ny=ny) #, wcs=AstropyWCS(self.wcs))
                self._mask = (result.array > 0.5).astype(float)
                self.wcs = self._rotate_wcs(self.mask, self.wcs, angle)
                self._wcs_header_merge()
                return result.array

            elif method == 'spline':
                from scipy.ndimage.interpolation import rotate as rt
                assert 0 < order <= 5 and isinstance(order, int), 'order of ' + method + ' must be within 0-5.'
                result = rt(self.mask, -angle, order=order, mode='constant', 
                            cval=cval, reshape=reshape)
                self._mask = (result > 0.5).astype(float)
                self.wcs = self._rotate_wcs(self.mask, self.wcs, angle)
                self._wcs_header_merge()
                return result
            elif method in ['bicubic', 'nearest','cubic','bilinear']:
                raise Warning("Cautious! Don't use ['bicubic', 'nearest', 'cubic', 'bilinear'] methods! They don't conserve the total flux!")
                try:
                    from scipy.misc import imrotate
                except:
                    raise ImportError('# Import `scipy.misc.imrotate` failed! This function may no longer be included in scipy!')
                result = imrotate(self.mask, -angle, interp=method)
                self._mask = (result > 0.5).astype(float)
                self.wcs = self._rotate_wcs(self.mask, self.wcs, angle)
                self._wcs_header_merge()
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
        '''Shift the image of Stack object. The WCS of image will also be changed.

        Parameters:
            dx, dy (float): shift distance (in pixel) along x (horizontal) and y (vertical). 
                Note that elements in one row has the same y but different x. 
                Example: dx = 2 is to shift the image "RIGHT", dy = 3 is to shift the image "UP".
            method (str): interpolation method. Use 'lanczos' or 'spline' or 'iraf'.
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
            # Begin shift
            assert (order > 0) and isinstance(order, int), 'order of ' + method + ' must be positive interger.'
            galimg = InterpolatedImage(Image(self.image, dtype=float), 
                                    scale=self.pixel_scale, x_interpolant=Lanczos(order))
            galimg = galimg.shift(dx=dx * self.pixel_scale, dy=dy * self.pixel_scale)
            result = galimg.drawImage(scale=self.pixel_scale, nx=nx, ny=ny)#, wcs=AstropyWCS(self.wcs))
            self._image = result.array
            # Change the WCS of image
            hdr = copy.deepcopy(self.header)
            hdr['CRPIX1'] += dx
            hdr['CRPIX2'] += dy
            self.header = hdr
            self.wcs = wcs.WCS(hdr)
            self._wcs_header_merge()
            return result.array
        elif method == 'iraf':
            try:
                from pyraf import iraf
            except:
                raise ImportError('# Import `iraf` failed! Please check if `pyraf` and `iraf` is installed!')
            self.save_to_fits('./_temp.fits', 'image')
            iraf.imdel('./_shift_temp.fits')
            iraf.imshift('./_temp.fits', './_shift_temp.fits', dx, dy, bound='constant', const=0.)
            hdu = fits.open('./_shift_temp.fits')
            self.image = hdu[0].data
            self.shape = hdu[0].data.shape
            self.header = hdu[0].header
            self.wcs = wcs.WCS(self.header)
            hdu.close()
            iraf.imdel('./*temp.fits')
        elif method == 'spline':
            from scipy.ndimage.interpolation import shift
            assert 0 < order <= 5 and isinstance(order, int), 'order of ' + method + ' must be within 0-5.'
            result = shift(self.image, [dy, dx], order=order, mode='constant', cval=cval)
            self._image = result
            # Change the WCS of image
            hdr = copy.deepcopy(self.header)
            hdr['CRPIX1'] -= dx
            hdr['CRPIX2'] += dy
            self.header = hdr
            self.wcs = wcs.WCS(hdr)
            self._wcs_header_merge()
            return result
        else:
            raise ValueError("# Not supported interpolation method. Use 'lanczos' or 'spline'.")

    def shift_mask(self, dx, dy, method='lanczos', order=5, cval=0.0):
        '''Shift the mask of Stack object.

        Parameters:
            dx, dy (float): shift distance (in pixel) along x (horizontal) and y (vertical). 
                Note that elements in one row has the same y but different x. 
                Example: dx = 2 is to shift the image "RIGHT", dy = 3 is to shift the image "UP".
            method (str): interpolation method. Use 'lanczos' or 'spline' or 'iraf'
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
            # Begin shift
            assert (order > 0) and isinstance(order, int), 'order of ' + method + ' must be positive interger.'
            galimg = InterpolatedImage(Image(self.mask, dtype=float), 
                                    scale=self.pixel_scale, x_interpolant=Lanczos(order))
            galimg = galimg.shift(dx=dx * self.pixel_scale, dy=dy * self.pixel_scale)
            result = galimg.drawImage(scale=self.pixel_scale, nx=nx, ny=ny)#, wcs=AstropyWCS(self.wcs))
            self._mask = (result.array > 0.5).astype(float)
            # Change the WCS of image
            hdr = copy.deepcopy(self.header)
            hdr['CRPIX1'] += dx
            hdr['CRPIX2'] += dy
            self.header = hdr
            self.wcs = wcs.WCS(hdr)
            self._wcs_header_merge()
            return result.array
        elif method == 'iraf':
            try:
                from pyraf import iraf
            except:
                raise ImportError('# Import `iraf` failed! Please check if `pyraf` and `iraf` is installed!')
            self.save_to_fits('./_temp.fits', 'mask')
            iraf.imdel('./_shift_temp.fits')
            iraf.imshift('./_temp.fits', './_shift_temp.fits', dx, dy, interpo='poly3', bound='constant', const=0.)
            hdu = fits.open('./_shift_temp.fits')
            self.mask = hdu[0].data
            self.shape = hdu[0].data.shape
            self.header = hdu[0].header
            self.wcs = wcs.WCS(self.header)
            hdu.close()
            iraf.imdel('./*temp.fits')
        elif method == 'spline':
            from scipy.ndimage.interpolation import shift
            assert 0 < order <= 5 and isinstance(order, int), 'order of ' + method + ' must be within 0-5.'
            result = shift(self.mask, [dy, dx], order=order, mode='constant', cval=cval)
            self._mask = (result > 0.5).astype(float)
            # Change the WCS of image
            hdr = copy.deepcopy(self.header)
            hdr['CRPIX1'] += dx
            hdr['CRPIX2'] += dy
            self.header = hdr
            self.wcs = wcs.WCS(hdr)
            self._wcs_header_merge()
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
    
    # Resize image/mask
    def _resize_wcs(self, img, w, f):
        w_temp = copy.deepcopy(w)
        ra_cen, dec_cen = w_temp.wcs_pix2world(img.shape[1]/2, img.shape[0]/2, 0)
        w_temp.wcs.crval = [ra_cen, dec_cen]
        w_temp.wcs.crpix = [img.shape[1]/2 * f, img.shape[0]/2 * f]
        #ra_cen, dec_cen = w_temp.wcs_pix2world(0, 0, 0)
        #w_temp.wcs.crval = [ra_cen, dec_cen]
        #w_temp.wcs.crpix = [0, 0]
        # move the reference pixel to (0, 0)
        w_temp.wcs.cd /= f
        return w_temp
    
    def _wcs_header_merge(self):
        """
        Look! this function must be used just before `return`! If you use it earlier, you'll make big trouble!
        """
        self.header['NAXIS1'] = self.image.shape[1]
        self.header['NAXIS2'] = self.image.shape[0]
        if self.wcs is not None:
            wcs_header = self.wcs.to_header()
            import fnmatch
            for i in self.header:
                if i in wcs_header:
                    self.header[i] = wcs_header[i]
                if fnmatch.fnmatch(i, 'CD?_?'):
                    self.header[i] = wcs_header['PC' + i.lstrip('CD')]
        self.wcs = wcs.WCS(self.header)


    def resize_image(self, f, method='lanczos', order=5, cval=0.0):
        '''Zoom/Resize the image of Stack object. 
        f > 1 means the image will be resampled! f < 1 means the image will be degraded.

        Cautious: don't use ['bicubic', 'nearest', 'cubic', 'bilinear'] methods! They don't conserve the total flux!

        Parameters:
            f (float): the positive factor of zoom. If 0 < f < 1, the image will be resized to smaller one.
            method (str): interpolation method. Use 'lanczos' or 'spline' or 'iraf'.
            order (int): the order of spline interpolation (within 0-5) or Lanczos interpolation (>0).
            cval (scalar): value to fill the edges. Default is NaN.

        Returns:
            shift_image: ndarray.
        '''
        if method == 'lanczos':
            try: # try to import galsim
                from galsim import degrees, Angle
                from galsim.interpolant import Lanczos
                from galsim import Image, InterpolatedImage
                from galsim.fitswcs import AstropyWCS
            except:
                raise ImportError('# Import `galsim` failed! Please check if `galsim` is installed!')

            assert (order > 0) and isinstance(order, int), 'order of ' + method + ' must be positive interger.'
            galimg = InterpolatedImage(Image(self.image, dtype=float), 
                                    scale=self.pixel_scale, x_interpolant=Lanczos(order))
            #galimg = galimg.magnify(f)
            ny, nx = self.image.shape
            result = galimg.drawImage(scale=self.pixel_scale / f, nx=round(nx * f), ny=round(ny * f))#, wcs=AstropyWCS(self.wcs))
            self.wcs = self._resize_wcs(self.image, self.wcs, f)
            self._image = result.array
            self.shape = self.image.shape
            self._wcs_header_merge()
            self.pixel_scale /= f
            return result.array
        elif method == 'iraf':
            try:
                from pyraf import iraf
            except:
                raise ImportError('# Import `iraf` failed! Please check if `pyraf` and `iraf` is installed!')
            self.save_to_fits('./_temp.fits', 'image')
            iraf.imdel('./_resize_temp.fits')
            if f > 1:
                iraf.magnify('./_temp.fits', './_resize_temp.fits', f, f, interpo='poly3', bound='constant', const=0.)
            else:
                iraf.blkavg('./_temp.fits', './_resize_temp.fits', 1/f, 1/f, option='sum')
            hdu = fits.open('./_resize_temp.fits')
            self.image = hdu[0].data
            self.shape = hdu[0].data.shape
            self.header = hdu[0].header
            self.wcs = wcs.WCS(self.header)
            self.pixel_scale /= f
            hdu.close()
            iraf.imdel('./*temp.fits')
        elif method == 'spline':
            from scipy.ndimage import zoom
            assert 0 < order <= 5 and isinstance(order, int), 'order of ' + method + ' must be within 0-5.'
            result = zoom(self.image, float(f), order=order, mode='constant', cval=cval)
            self.wcs = self._resize_wcs(self.image, self.wcs, f)
            self._image = result
            self.shape = self.image.shape
            self._wcs_header_merge()
            self.pixel_scale /= f
            return result
        elif method in ['bicubic', 'nearest', 'cubic', 'bilinear']:
            raise Warning("Cautious! Don't use ['bicubic', 'nearest', 'cubic', 'bilinear'] methods! They don't conserve the total flux!")
            try:
                from scipy.misc import imresize
            except:
                raise ImportError('# Import `scipy.misc.imresize` failed! This function may no longer be included in scipy!')
            result = imresize(self.image, float(f), interp=method)
            self.wcs = self._resize_wcs(self.image, self.wcs, f)
            self._image = result.astype(float)
            self.shape = self.image.shape
            self._wcs_header_merge()
            self.pixel_scale /= f
            return result.astype(float)
        else:
            raise ValueError("# Not supported interpolation method. Use 'lanczos' or 'spline'.")
    
    def resize_mask(self, f, method='lanczos', order=5, cval=0.0):
        '''Zoom/Resize the mask of Stack object.
        Cautious: don't use ['bicubic', 'nearest', 'cubic', 'bilinear'] methods! They don't conserve the total flux!

        Parameters:
            f (float): the positive factor of zoom. If 0 < f < 1, the image will be resized to smaller one.
            method (str): interpolation method. Use 'lanczos' or 'spline' or 'iraf'.
            order (int): the order of spline interpolation (within 0-5) or Lanczos interpolation (>0).
            cval (scalar): value to fill the edges. Default is NaN.

        Returns:
            shift_image: ndarray.
        '''
        if method == 'lanczos':
            try: # try to import galsim
                from galsim import degrees, Angle
                from galsim.interpolant import Lanczos
                from galsim import Image, InterpolatedImage
                from galsim.fitswcs import AstropyWCS
            except:
                raise ImportError('# Import `galsim` failed! Please check if `galsim` is installed!')

            assert (order > 0) and isinstance(order, int), 'order of ' + method + ' must be positive interger.'
            galimg = InterpolatedImage(Image(self.mask, dtype=float), 
                                    scale=self.pixel_scale, x_interpolant=Lanczos(order))
            #galimg = galimg.magnify(f)
            ny, nx = self.mask.shape
            result = galimg.drawImage(scale=self.pixel_scale / f, nx=round(nx * f), ny=round(ny * f))#, wcs=AstropyWCS(self.wcs))
            self.wcs = self._resize_wcs(self.mask, self.wcs, f)
            self._mask = result.array
            self.shape = self.mask.shape
            self._wcs_header_merge()
            self.pixel_scale /= f
            return result.array
        elif method == 'iraf':
            try:
                from pyraf import iraf
            except:
                raise ImportError('# Import `iraf` failed! Please check if `pyraf` and `iraf` is installed!')
            self.save_to_fits('./_temp.fits', 'mask')
            iraf.imdel('./_resize_temp.fits')
            iraf.magnify('./_temp.fits', './_resize_temp.fits', f, f, interpo='poly3', bound='constant', const=0.)
            hdu = fits.open('./_resize_temp.fits')
            self.mask = hdu[0].data
            self.shape = hdu[0].data.shape
            self.header = hdu[0].header
            self.wcs = wcs.WCS(self.header)
            self.pixel_scale /= f
            hdu.close()
            iraf.imdel('./*temp.fits')
        elif method == 'spline':
            from scipy.ndimage import zoom
            assert 0 < order <= 5 and isinstance(order, int), 'order of ' + method + ' must be within 0-5.'
            result = zoom(self.mask, float(f), order=order, mode='constant', cval=cval)
            self._mask = result
            self.wcs = self._resize_wcs(self.mask, self.wcs, f)
            self.shape = self.mask.shape
            self._wcs_header_merge()
            self.pixel_scale /= f
            return result
        elif method in ['bicubic', 'nearest', 'cubic', 'bilinear']:
            raise Warning("Cautious! Don't use ['bicubic', 'nearest', 'cubic', 'bilinear'] methods! They don't conserve the total flux!")
            try:
                from scipy.misc import imresize
            except:
                raise ImportError('# Import `scipy.misc.imresize` failed! This function may no longer be included in scipy!')
            result = imresize(self.mask, float(f), interp=method)
            self._mask = result.astype(float)
            self.wcs = self._resize_wcs(self.mask, self.wcs, f)
            self.shape = self.mask.shape
            self._wcs_header_merge()
            self.pixel_scale /= f
            return result.astype(float)
        else:
            raise ValueError("# Not supported interpolation method. Use 'lanczos' or 'spline'.")

    def resize_Stack(self, f, method='lanczos', order=5, cval=0.0):
        '''Resize the Stack object. f > 1 means the image will be resampled! f < 1 means the image will be degraded.

        Parameters:
            angle (float): rotation angle in degress, counterclockwise.
            order (int): the order of spline interpolation, can be in the range 0-5.
            reshape (bool): if True, the output shape is adapted so that the rorated image 
                is contained completely in the output array.
            cval (scalar): value to fill the edges. Default is NaN.
        
        Returns:
        '''
        self.resize_image(f, method=method, order=order, cval=cval)
        if hasattr(self, 'mask'):
            self.resize_mask(f, method=method, order=order, cval=cval)


    # Display image/mask
    def display_image(self, **kwargs):
        display_single(self.image, scale_bar_length=self.scale_bar_length, **kwargs)
    def display_mask(self, **kwargs):
        display_single(self.mask, scale='linear', 
                        cmap=SEG_CMAP, scale_bar_length=self.scale_bar_length, **kwargs)
    def display_Stack(self, **kwargs):
        if hasattr(self, 'mask'):
            display_single(self.image * (~self.mask.astype(bool)), 
                            scale_bar_length=self.scale_bar_length, **kwargs)
        else:
            self.display_image()
       

class StackSky(Stack):
    def __init__(self, img, header, skyobj, mask=None, aper_name='aper57'):
        Stack.__init__(self, img, mask, header=header)
        self.name = 'sky'
        self.scale_bar_length = 1
        from unagi.sky import SkyObjs, AperPhot, S18A_APER, S18A_APER_ID
        cutout, cen_pos, cutout_header = img_cutout(self.image, self.wcs, skyobj['ra'], skyobj['dec'], 
                                         size=2 * S18A_APER[aper_name].r_arcsec, save=False)
        self._image = cutout.data
        self.shape = self.image.shape
        self.wcs = cutout.wcs
        self.header = cutout_header
        self.cen_xy = cen_pos[0]
        self.dx = cen_pos[1]
        self.dy = cen_pos[2]
        

        if hasattr(self, 'mask'):
            cutout, _, _ = img_cutout(self.mask, self.wcs, skyobj['ra'], skyobj['dec'], 
                                   size=2 * S18A_APER[aper_name].r_arcsec, save=False)
            self._mask = cutout.data

    def centralize(self, method='lanczos', order=5, cval=0.0):
        self.shift_Stack(self.dx, self.dy, method=method, order=order, cval=cval)
    
    def get_masked_image(self, cval=np.nan):
        if not hasattr(self, 'mask'):
            raise Warning("This `StackSky` object doesn't have a `mask`!")
            return self.image
        else:
            imgcp = copy.copy(self.image)
            imgcp[self.mask.astype(bool)] = cval
            return imgcp
            #return self.image * (~self.mask.astype(bool))
        
    
class StackStar(Stack):
    def __init__(self, img, header, starobj, halosize=40, padsize=40, mask=None, hscmask=None):
        """Halosize is the radius!!!
        RA, DEC are not supported yet!
        """
        Stack.__init__(self, img, mask, header=header)
        #if hscmask is not None:
        self.hscmask = hscmask
        self.name = 'star'
        self.scale_bar_length = 3
        # Trim the image to star size
        # starobj should at least contain x, y, (or ra, dec) and 
        # Position of a star, in numpy convention
        x_int = int(starobj['x'])
        y_int = int(starobj['y'])
        dx = -1.0 * (starobj['x'] - x_int)
        dy = -1.0 * (starobj['y'] - y_int)
        halosize = int(halosize)
        # Make padded image to deal with stars near the edges
        padsize = int(padsize)
        ny, nx = self.image.shape
        im_padded = np.zeros((ny + 2 * padsize, nx + 2 * padsize))
        im_padded[padsize: ny + padsize, padsize: nx + padsize] = self.image
        # Star itself, but no shift here.
        halo = (im_padded[y_int + padsize - halosize: y_int + padsize + halosize + 1, 
                          x_int + padsize - halosize: x_int + padsize + halosize + 1])
        self._image = halo
        self.shape = halo.shape
        self.cen_xy = [x_int, y_int]
        self.dx = dx
        self.dy = dy   
        # FLux
        self.flux = starobj['flux']
        self.fluxann = starobj['flux_ann']

        if hasattr(self, 'mask'):
            im_padded = np.zeros((ny + 2 * padsize, nx + 2 * padsize))
            im_padded[padsize: ny + padsize, padsize: nx + padsize] = self.mask
            # Mask itself, but no shift here.
            halo = (im_padded[y_int + padsize - halosize: y_int + padsize + halosize + 1, 
                              x_int + padsize - halosize: x_int + padsize + halosize + 1])
            self._mask = halo
        
        if hasattr(self, 'hscmask'):
            im_padded = np.zeros((ny + 2 * padsize, nx + 2 * padsize))
            im_padded[padsize: ny + padsize, padsize: nx + padsize] = self.hscmask
            # Mask itself, but no shift here.
            halo = (im_padded[y_int + padsize - halosize: y_int + padsize + halosize + 1, 
                              x_int + padsize - halosize: x_int + padsize + halosize + 1])
            self.hscmask = halo

    def centralize(self, method='lanczos', order=5, cval=0.0):
        self.shift_Stack(self.dx, self.dy, method=method, order=order, cval=cval)

    def mask_out_contam(self, method='hscmask', blowup=True, show_fig=True, verbose=True):
        if method == 'hscmask':
            from unagi import mask
            from .image import mask_remove_cen_obj
            detect_mask = mask.Mask(self.hscmask, data_release='s18a').extract('DETECTED').astype(float)
            detect_mask = mask_remove_cen_obj(detect_mask)
            if blowup is True:
                from astropy.convolution import convolve, Gaussian2DKernel
                cv = convolve(detect_mask, Gaussian2DKernel(1.5))
                detect_mask = (cv > 0.1).astype(float)
            self.mask = detect_mask
            return
        else: # method = 'sep'
            from astropy.convolution import convolve, Box2DKernel
            from .image import extract_obj, seg_remove_cen_obj
            img_blur = convolve(abs(self.image), Box2DKernel(2))
            img_objects, img_segmap = extract_obj(abs(img_blur), b=5, f=4, sigma=5.5, minarea=5, pixel_scale=self.pixel_scale,
                                                  deblend_nthresh=32, deblend_cont=0.05, 
                                                  sky_subtract=False, show_fig=show_fig, verbose=verbose)
            # remove central object from segmap
            img_segmap = seg_remove_cen_obj(img_segmap) 
            detect_mask = (img_segmap != 0).astype(float)
            if blowup is True:
                from astropy.convolution import convolve, Gaussian2DKernel
                cv = convolve(detect_mask, Gaussian2DKernel(1.5))
                detect_mask_1 = (cv > 0.05)
            img_blur = abs(img_blur) 
            img_blur[detect_mask_1] = 0
            # Second time
            img_objects, img_segmap = extract_obj(img_blur, b=5, f=4, sigma=4.5, minarea=5, pixel_scale=self.pixel_scale,
                                                  deblend_nthresh=32, deblend_cont=0.0001, 
                                                  sky_subtract=False, show_fig=show_fig, verbose=verbose)
            # remove central object from segmap
            img_segmap = seg_remove_cen_obj(img_segmap) 
            detect_mask = (img_segmap != 0).astype(float)
            if blowup is True:
                from astropy.convolution import convolve, Gaussian2DKernel
                cv = convolve(detect_mask, Gaussian2DKernel(1.5))
                detect_mask_2 = (cv > 0.05)
            self.mask = (detect_mask_1 + detect_mask_2).astype(float)
            return 

    def mask_out_contam_bright(self, method='hscmask', blowup=True, show_fig=True, verbose=True):
        if method == 'hscmask':
            from unagi import mask
            from .image import mask_remove_cen_obj
            detect_mask = mask.Mask(self.hscmask, data_release='s18a').extract('DETECTED').astype(float)
            detect_mask = mask_remove_cen_obj(detect_mask)
            if blowup is True:
                from astropy.convolution import convolve, Gaussian2DKernel
                cv = convolve(detect_mask, Gaussian2DKernel(1.5))
                detect_mask = (cv > 0.1).astype(float)
            self.mask = detect_mask
            return
        else: # method = 'sep'
            from astropy.convolution import convolve, Box2DKernel
            from .image import extract_obj, seg_remove_cen_obj
            img_blur = convolve(abs(self.image), Box2DKernel(2))
            img_objects, img_segmap = extract_obj(abs(img_blur), b=5, f=4, sigma=4.5, minarea=5, pixel_scale=self.pixel_scale,
                                                  deblend_nthresh=32, deblend_cont=0.005, 
                                                  sky_subtract=False, show_fig=show_fig, verbose=verbose)
            # remove central object from segmap
            img_segmap = seg_remove_cen_obj(img_segmap) 
            detect_mask = (img_segmap != 0).astype(float)
            if blowup is True:
                from astropy.convolution import convolve, Gaussian2DKernel
                cv = convolve(detect_mask, Gaussian2DKernel(1.5))
                detect_mask_1 = (cv > 0.05)
            self.mask = (detect_mask_1).astype(float)
            return 


    def sub_bkg(self, verbose=True):
        # Here I subtract local sky background
        # Evaluate local sky backgroud within `halo_i`
        # Actually this should be estimated in larger cutuouts.
        # So make another cutout (larger)!
        from astropy.convolution import convolve, Box2DKernel
        from .image import extract_obj, seg_remove_cen_obj
        from sep import Background
        img_blur = convolve(abs(self.image), Box2DKernel(2))
        img_objects, img_segmap = extract_obj(abs(img_blur), b=5, f=4, sigma=4.5, minarea=2, pixel_scale=self.pixel_scale,
                                                deblend_nthresh=32, deblend_cont=0.0001, 
                                                sky_subtract=False, show_fig=False, verbose=False)
        bk = Background(self.image, img_segmap != 0)
        glbbck = bk.globalback
        self.globalback = glbbck
        if verbose:
            print('# Global background: ', glbbck)
        self.image -= glbbck

    def get_masked_image(self, cval=np.nan):
        if not hasattr(self, 'mask'):
            raise Warning("This `StackStar` object doesn't have a `mask`!")
            return self.image
        else:
            imgcp = copy.copy(self.image)
            imgcp[self.mask.astype(bool)] = cval
            return imgcp

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

