# -*- coding: utf-8 -*-
# Import packages
import warnings
warnings.simplefilter('ignore')

# Version
__version__ = "0.0.0"
__name__ = 'lvhuo'

# Define pixel scale of different surveys, unit = arcsec / pixel
HSC_pixel_scale = 0.168
DECaLS_pixel_scale = 0.262
Dragonfly_pixel_scale = 2.5
SDSS_pixel_scale = 0.395
CFHT_pixel_scale = 0.3715

# Define zeropoint of different surveys
HSC_zeropoint = 27.0
DECaLS_zeropoint = 22.5
SDSS_zeropoint = 22.5
CFHT_zeropoint = 30.0

# Star catalogs in VizieR
USNO_vizier = 'I/252/out'
APASS_vizier = 'II/336'

HSC_binray_mask_dict = {0: 'BAD',
                        1:  'SAT (saturated)',
                        2:  'INTRP (interpolated)',
                        3:  'CR (cosmic ray)',
                        4:  'EDGE (edge of the CCD)',
                        5:  'DETECTED',
                        6:  'DETECTED_NEGATIVE',
                        7:  'SUSPECT (suspicious pixel)',
                        8:  'NO_DATA',
                        9:  'BRIGHT_OBJECT (bright star mask, not available in S18A yet)',
                        10: 'CROSSTALK', 
                        11: 'NOT_DEBLENDED (For objects that are too big to run deblender)',
                        12: 'UNMASKEDNAN',
                        13: 'REJECTED',
                        14: 'CLIPPED',
                        15: 'SENSOR_EDGE',
                        16: 'INEXACT_PSF'}

SkyObj_aperture_dic = { '20': 5.0,
                        '30': 9.0,
                        '40': 12.0,
                        '57': 17.0,
                        '84': 25.0,
                        '118': 35.0 }