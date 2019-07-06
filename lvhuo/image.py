import os
import sep
import copy
import scipy

import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column
import astropy.units as u
from astropy.coordinates import SkyCoord
from lvhuo import USNO_vizier, APASS_vizier
from .display import display_single, SEG_CMAP

# Simply remove stars by masking them out
def query_star(img, header, method='gaia', bright_lim=15.5, catalog_dir=None):
    """ 
    Parameters:
        img (2-D numpy array): image itselt.
        header: the header of this image.
        method (str): here three methods are provided: 'gaia', 'apass' or 'usno'.
        bright_lim (float): the magnitude limit of stars to be masked out. 
        catalog_dir (str): optional, you can provide local catalog here.

    Returns:
        star_cat
    """
    if method.lower() == 'gaia':
        from kungpao import imtools, query
        from astropy import wcs
        print('### Querying Gaia Data ###')
        gaia_stars, gaia_mask = imtools.gaia_star_mask(img, wcs.WCS(header), 
                                                       pix=header['CD2_2'] * 3600, 
                                                       size_buffer=4, gaia_bright=bright_lim, 
                                                       factor_f=2.0, factor_b=1.2)
        return gaia_stars
    elif method.lower() == 'apass' or method.lower() == 'usno':
        if catalog_dir is not None: # catalog is provided
            print("You provided star catalog file!")
            # Read catalog directory
            _, file_extension = os.path.splitext(catalog_dir)
            if file_extension.lower() == 'fits':
                catalog = Table.read(catalog_dir, format='fits')
            else:
                catalog = Table.read(catalog_dir, format='ascii')
        else: # Online query!
            print("### Online querying " + method.upper() + " data from VizieR. ###")
            # Construct query
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            from astropy import wcs
            w = wcs.WCS(header)
            c1 = SkyCoord(float(w.wcs_pix2world(0, 0, 0)[0])*u.degree, 
                          float(w.wcs_pix2world(0, 0, 0)[1])*u.degree, 
                          frame='icrs')
            c2 = SkyCoord(float(w.wcs_pix2world(img.shape[1], img.shape[0], 0)[0])*u.degree, 
                          float(w.wcs_pix2world(img.shape[1], img.shape[0], 0)[1])*u.degree, 
                          frame='icrs')
            c_cen = SkyCoord(float(w.wcs_pix2world(img.shape[1]//2, img.shape[0]//2, 0)[0])*u.degree, 
                             float(w.wcs_pix2world(img.shape[1]//2, img.shape[0]//2, 0)[1])*u.degree, 
                             frame='icrs')
            radius = c1.separation(c2).to(u.degree).value
            from astroquery.vizier import Vizier
            from astropy.coordinates import Angle
            Vizier.ROW_LIMIT = -1
            if method.lower() == 'apass':
                query_method = APASS_vizier
            elif method.lower() == 'usno':
                query_method = USNO_vizier
            else:
                raise ValueError("Method must be 'gaia', 'apass' or 'usno'!")
            result = Vizier.query_region(str(c_cen.ra.value) + ' ' + str(c_cen.dec.value), 
                                         radius=Angle(radius, "deg"), catalog=query_method)
            catalog = result.values()[0]
            catalog.rename_column('RAJ2000', 'ra')
            catalog.rename_column('DEJ2000', 'dec')
            if method.lower() == 'apass':
                catalog.rename_column('e_RAJ2000', 'e_ra')
                catalog.rename_column('e_DEJ2000', 'e_dec')
        return catalog
    else:
        raise ValueError("Method must be 'gaia', 'apass' or 'usno'!")
        return 

def circularize(img, n=14, print_g=True):
    from scipy.ndimage.interpolation import rotate
    a = img
    for i in range(n):
        theta = 360 / 2**(i + 1)
        if i == 0:
            temp = a
        else:
            temp = b
        b = rotate(temp, theta, order=3, mode='constant', cval=0.0, reshape=False)
        c = .5 * (a + b)
        a = b
        b = c 
    if print_g is True:
        print('The asymmetry parameter g of given image is ' + 
                str(abs(np.sum(b - img))))
    return b

# Cutout image
def img_cutout(img, wcs, coord_1, coord_2, size=60.0, pix=0.168,
               prefix='img_cutout', pixel_unit=False, img_header=None, 
               out_dir=None, save=True):
    """(From kungpao) Generate image cutout with updated WCS information.
    ----------
    Parameters:
        pixel_unit: boolen, optional
                    When True, coord_1, cooord_2 becomes X, Y pixel coordinates.
                    Size will also be treated as in pixels.
        img: 2d array.
        wcs: astropy wcs object of the input image.
        coord_1: ra of the center.
        coord_2: dec of the center.
        size: image size, default in arcsec unit.
        pix: pixel size.
        img_header: the astropy header object of the input image. 
                    In case you can save the infomation in this header to the new header.
    """
    from astropy.nddata import Cutout2D
    if not pixel_unit:
        # imgsize in unit of arcsec
        cutout_size = np.asarray(size) / pix
        cen_x, cen_y = wcs.wcs_world2pix(coord_1, coord_2, 0)
    else:
        cutout_size = np.asarray(size)
        cen_x, cen_y = coord_1, coord_2

    cen_pos = (int(cen_x), int(cen_y))
    dx = -1.0 * (cen_x - int(cen_x))
    dy = -1.0 * (cen_y - int(cen_y))

    # Generate cutout
    cutout = Cutout2D(img, cen_pos, cutout_size, wcs=wcs)

    # Update the header
    cutout_header = cutout.wcs.to_header()
    if img_header is not None:
        intersect = [k for k in img_header if k not in cutout_header]
        for keyword in intersect:
            cutout_header.set(keyword, img_header[keyword], img_header.comments[keyword])
    
    # Build a HDU
    hdu = fits.PrimaryHDU(header=cutout_header)
    hdu.data = cutout.data

    # Save FITS image
    if save:
        fits_file = prefix + '.fits'
        if out_dir is not None:
            fits_file = os.path.join(out_dir, fits_file)

        hdu.writeto(fits_file, overwrite=True)

    return cutout, [cen_pos, dx, dy]

# evaluate_sky objects for a given image
def extract_obj(img, b=30, f=5, sigma=5, pixel_scale=0.168, minarea=5, 
    deblend_nthresh=32, deblend_cont=0.005, clean_param=1.0, 
    sky_subtract=False, show_fig=True, verbose=True, flux_auto=True, flux_aper=None):
    '''Extract objects for a given image, using `sep`. This is from `slug`.

    Parameters:
    ----------
    img: 2-D numpy array
    b: float, size of box
    f: float, size of convolving kernel
    sigma: float, detection threshold
    pixel_scale: float

    Returns:
    -------
    objects: astropy Table, containing the positions,
        shapes and other properties of extracted objects.
    segmap: 2-D numpy array, segmentation map
    '''

    # Subtract a mean sky value to achieve better object detection
    b = 30  # Box size
    f = 5   # Filter width
    bkg = sep.Background(img, bw=b, bh=b, fw=f, fh=f)
    data_sub = img - bkg.back()
    
    sigma = sigma
    if sky_subtract:
        input_data = data_sub
    else:
        input_data = img
    objects, segmap = sep.extract(input_data,
                                  sigma,
                                  err=bkg.globalrms,
                                  segmentation_map=True,
                                  filter_type='matched',
                                  deblend_nthresh=deblend_nthresh,
                                  deblend_cont=deblend_cont,
                                  clean=True,
                                  clean_param=clean_param,
                                  minarea=minarea)
    if verbose:                              
        print("# Detect %d objects" % len(objects))
    objects = Table(objects)
    objects.add_column(Column(data=np.arange(len(objects)) + 1, name='index'))
    # Maximum flux, defined as flux within six 'a' in radius.
    objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], 
                                    6. * objects['a'])[0], name='flux_max'))
    # Add FWHM estimated from 'a' and 'b'. 
    # This is suggested here: https://github.com/kbarbary/sep/issues/34
    objects.add_column(Column(data=2* np.sqrt(np.log(2) * (objects['a']**2 + objects['b']**2)), 
                              name='fwhm_custom'))
    
    # Use Kron radius to calculate FLUX_AUTO in SourceExtractor.
    # Here PHOT_PARAMETER = 2.5, 3.5
    if flux_auto:
        kronrad, krflag = sep.kron_radius(input_data, objects['x'], objects['y'], 
                                          objects['a'], objects['b'], 
                                          objects['theta'], 6.0)
        flux, fluxerr, flag = sep.sum_circle(input_data, objects['x'], objects['y'], 
                                            2.5 * (kronrad), subpix=1)
        flag |= krflag  # combine flags into 'flag'

        r_min = 1.75  # minimum diameter = 3.5
        use_circle = kronrad * np.sqrt(objects['a'] * objects['b']) < r_min
        cflux, cfluxerr, cflag = sep.sum_circle(input_data, objects['x'][use_circle], objects['y'][use_circle],
                                                r_min, subpix=1)
        flux[use_circle] = cflux
        fluxerr[use_circle] = cfluxerr
        flag[use_circle] = cflag
        objects.add_column(Column(data=flux, name='flux_auto'))
        objects.add_column(Column(data=kronrad, name='kron_rad'))
        
    if flux_aper is not None:
        objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], flux_aper[0])[0], 
                                  name='flux_aper_1'))
        objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], flux_aper[1])[0], 
                                  name='flux_aper_2')) 
        objects.add_column(Column(data=sep.sum_circann(input_data, objects['x'], objects['y'], 
                                       flux_aper[0], flux_aper[1])[0], name='flux_ann'))
        '''
        objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], flux_aper[0] * objects['a'])[0], 
                                  name='flux_aper_1'))
        objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], flux_aper[1] * objects['a'])[0], 
                                  name='flux_aper_2')) 
        objects.add_column(Column(data=sep.sum_circann(input_data, objects['x'], objects['y'], 
                                       flux_aper[0] * objects['a'], flux_aper[1] * objects['a'])[0], name='flux_ann'))
        '''

    # plot background-subtracted image
    if show_fig:
        fig, ax = plt.subplots(1,2, figsize=(12,6))

        ax[0] = display_single(data_sub, ax=ax[0], scale_bar=False, pixel_scale=pixel_scale)
        from matplotlib.patches import Ellipse
        # plot an ellipse for each object
        for obj in objects:
            e = Ellipse(xy=(obj['x'], obj['y']),
                        width=8*obj['a'],
                        height=8*obj['b'],
                        angle=obj['theta'] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax[0].add_artist(e)
        ax[1] = display_single(segmap, scale='linear', cmap=SEG_CMAP , ax=ax[1])
    return objects, segmap


def seg_remove_cen_obj(seg):
    """Remove the central object from the segmentation.
    Parameters:
        seg (numpy 2-D array): segmentation map

    Returns:
        seg_copy (numpy 2-D array): the segmentation map with central object removed
    """
    seg_copy = copy.deepcopy(seg)
    seg_copy[seg == seg[int(seg.shape[0] / 2.0), int(seg.shape[1] / 2.0)]] = 0

    return seg_copy

def mask_remove_cen_obj(mask):
    """Remove the central object from the binary 0-1 mask.
    Parameters:
        mask (numpy 2-D array): binary mask

    Returns:
        mask_copy (numpy 2-D array): a mask with central object removed
    """
    from scipy.ndimage import label
    mask_copy = copy.deepcopy(mask)
    seg = label(mask)[0]
    mask_copy[seg == seg[int(seg.shape[0] / 2.0), int(seg.shape[1] / 2.0)]] = 0

    return mask_copy

def seg_remove_obj(seg, x, y):
    """Remove an object from the segmentation given its coordinate.
        
    Parameters:
        seg (numpy 2-D array): segmentation mask
        x, y (int): coordinates.
    Returns:
        seg_copy (numpy 2-D array): the segmentation map with certain object removed
    """
    seg_copy = copy.deepcopy(seg)
    seg_copy[seg == seg[int(y), int(x)]] = 0

    return seg_copy

def mask_remove_obj(mask):
    """Remove an object from the mask given its coordinate.
        
    Parameters:
        mask (numpy 2-D array): binary mask
        x, y (int): coordinates.
    Returns:
        mask_copy (numpy 2-D array): the mask with certain object removed
    """
    from scipy.ndimage import label
    mask_copy = copy.deepcopy(mask)
    seg = label(mask)[0]
    mask_copy[seg == seg[int(y), int(x)]] = 0

    return mask_copy

# Save 2-D numpy array to `fits`
def save_to_fits(img, fits_file, wcs=None, header=None, overwrite=True):
    """Save numpy 2-D arrays to `fits` file. (from `kungpao`)
    Parameters:
        img (np.array, 2d): The 2-D array to be saved
        fits_file (str): File name of `fits` file
        wcs (astropy.wcs.WCS class): World coordinate system of this image
        header (astropy.io.fits.header or str): header of this image
        overwrite (bool): Default is True

    Returns:
        None
    """
    if wcs is not None:
        wcs_header = wcs.to_header()
        img_hdu = fits.PrimaryHDU(img, header=wcs_header)
    else:
        img_hdu = fits.PrimaryHDU(img)

    if header is not None:
        img_hdu.header = header

    if os.path.islink(fits_file):
        os.unlink(fits_file)

    img_hdu.writeto(fits_file, overwrite=overwrite)

    return


def psf_SBP(psf_path, msk_path, pixel_scale, iraf_path, step=0.10, 
            sma_ini=10.0, sma_max=100.0, n_clip=3, maxTry=5, low_clip=3.0, upp_clip=2.5, 
            outPre=None, verbose=True):
    #from kungpao.galsbp import galSBP
    from compsub.galSBP import galSBP
    hdu = fits.open(psf_path)
    psf = hdu[0].data
    x_cen = psf.shape[1] // 2 + 1
    y_cen = psf.shape[0] // 2 + 1
    hdu.close()

    ISO = iraf_path + 'x_isophote.e'
    TBL = iraf_path + 'x_ttools.e'
    integrade_mode = 'median'   # or 'mean', or 'bi-linear'
    ell_fix, _ = galSBP(
        psf_path,
        mask=msk_path,
        galX=x_cen,
        galY=y_cen,
        galQ=1.0,
        galPA=0.0,
        iniSma=sma_ini,
        minSma=0.0,
        maxSma=psf.shape[0]/2,
        pix=1 / pixel_scale,
        zpPhoto=0,
        expTime=0,
        stage=3,
        ellipStep=step,
        isophote=ISO,
        xttools=TBL,
        uppClip=upp_clip,
        lowClip=low_clip,
        nClip=n_clip,
        maxTry=5,
        fracBad=0.8,
        maxIt=300,
        recenter=False,
        harmonics="none",
        intMode=integrade_mode,
        saveOut=True,
        plMask=True,
        verbose=verbose,
        savePng=False,
        updateIntens=False,
        saveCsv=True,
        suffix='',
        location='./temp/',
        outPre=outPre + '-ellip-fix')
    return ell_fix
