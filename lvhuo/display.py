# Import packages
from __future__ import division, print_function

import numpy as np

from astropy.io import fits
from astropy import wcs
from astropy.table import Table
from astropy.visualization import (ZScaleInterval,
                                   AsymmetricPercentileInterval)
from astropy.visualization import make_lupton_rgb

from matplotlib import colors
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter, MaxNLocator, FormatStrFormatter, AutoMinorLocator
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from palettable.colorbrewer.sequential import (Greys_9,
                                               OrRd_9,
                                               Blues_9,
                                               Purples_9,
                                               YlGn_9)

def random_cmap(ncolors=256, background_color='white'):
    """Random color maps.

    Generate a matplotlib colormap consisting of random (muted) colors.
    A random colormap is very useful for plotting segmentation images.

    Parameters
    ----------
    ncolors : int, optional
        The number of colors in the colormap.  The default is 256.
    random_state : int or `~numpy.random.RandomState`, optional
        The pseudo-random number generator state used for random
        sampling.  Separate function calls with the same
        ``random_state`` will generate the same colormap.

    Returns
    -------
    cmap : `matplotlib.colors.Colormap`
        The matplotlib colormap with random colors.

    Notes
    -----
    Based on: colormaps.py in photutils

    """
    prng = np.random.mtrand._rand

    h = prng.uniform(low=0.0, high=1.0, size=ncolors)
    s = prng.uniform(low=0.2, high=0.7, size=ncolors)
    v = prng.uniform(low=0.5, high=1.0, size=ncolors)

    hsv = np.dstack((h, s, v))
    rgb = np.squeeze(colors.hsv_to_rgb(hsv))

    if background_color is not None:
        if background_color not in colors.cnames:
            raise ValueError('"{0}" is not a valid background color '
                             'name'.format(background_color))
        rgb[0] = colors.hex2color(colors.cnames[background_color])

    return colors.ListedColormap(rgb)

# About the Colormaps
IMG_CMAP = plt.get_cmap('viridis')
IMG_CMAP.set_bad(color='black')
SEG_CMAP = random_cmap(ncolors=512, background_color=u'white')
SEG_CMAP.set_bad(color='white')
SEG_CMAP.set_under(color='white')

# Color map
BLK = Greys_9.mpl_colormap
ORG = OrRd_9.mpl_colormap
BLU = Blues_9.mpl_colormap
GRN = YlGn_9.mpl_colormap
PUR = Purples_9.mpl_colormap

def display_single(img,
                   pixel_scale=0.168,
                   physical_scale=None,
                   xsize=8,
                   ysize=8,
                   ax=None,
                   stretch='arcsinh',
                   scale='zscale',
                   contrast=0.25,
                   no_negative=False,
                   lower_percentile=1.0,
                   upper_percentile=99.0,
                   cmap=IMG_CMAP,
                   scale_bar=True,
                   scale_bar_length=5.0,
                   scale_bar_fontsize=20,
                   scale_bar_y_offset=0.5,
                   scale_bar_color='w',
                   scale_bar_loc='left',
                   color_bar=False,
                   color_bar_loc=1,
                   color_bar_width='75%',
                   color_bar_height='5%',
                   color_bar_fontsize=18,
                   color_bar_color='w',
                   add_text=None,
                   text_fontsize=30,
                   text_y_offset=0.80,
                   text_color='w'):
    """Display single image. From `kungpao`.

    Parameters
    ----------
        img: np 2-D array for image

        xsize: int, default = 8
            Width of the image.

        ysize: int, default = 8
            Height of the image.

    """
    if ax is None:
        fig = plt.figure(figsize=(xsize, ysize))
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax

    # Stretch option
    if stretch.strip() == 'arcsinh':
        img_scale = np.arcsinh(img)
    elif stretch.strip() == 'log':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log(img)
    elif stretch.strip() == 'log10':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log10(img)
    elif stretch.strip() == 'linear':
        img_scale = img
    else:
        raise Exception("# Wrong stretch option.")

    # Scale option
    if scale.strip() == 'zscale':
        try:
            zmin, zmax = ZScaleInterval(contrast=contrast).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    elif scale.strip() == 'percentile':
        try:
            zmin, zmax = AsymmetricPercentileInterval(
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    else:
        zmin, zmax = np.nanmin(img_scale), np.nanmax(img_scale)
    
    show = ax1.imshow(img_scale, origin='lower', cmap=cmap,
                      vmin=zmin, vmax=zmax)

    # Hide ticks and tick labels
    ax1.tick_params(
        labelbottom=False,
        labelleft=False,
        axis=u'both',
        which=u'both',
        length=0)
    #ax1.axis('off')

    # Put scale bar on the image
    (img_size_x, img_size_y) = img.shape
    if physical_scale is not None:
        pixel_scale *= physical_scale
    if scale_bar:
        if scale_bar_loc == 'left':
            scale_bar_x_0 = int(img_size_x * 0.04)
            scale_bar_x_1 = int(img_size_x * 0.04 +
                                (scale_bar_length / pixel_scale))
        else:
            scale_bar_x_0 = int(img_size_x * 0.95 -
                                (scale_bar_length / pixel_scale))
            scale_bar_x_1 = int(img_size_x * 0.95)

        scale_bar_y = int(img_size_y * 0.10)
        scale_bar_text_x = (scale_bar_x_0 + scale_bar_x_1) / 2
        scale_bar_text_y = (scale_bar_y * scale_bar_y_offset)
        if physical_scale is not None:
            if scale_bar_length > 1000:
                scale_bar_text = r'$%d\ \mathrm{Mpc}$' % int(scale_bar_length / 1000)
            else:
                scale_bar_text = r'$%d\ \mathrm{kpc}$' % int(scale_bar_length)
        else:
            if scale_bar_length < 60:
                scale_bar_text = r'$%d^{\prime\prime}$' % int(scale_bar_length)
            elif 60 < scale_bar_length < 3600:
                scale_bar_text = r'$%d^{\prime}$' % int(scale_bar_length / 60)
            else: 
                scale_bar_text = r'$%d^{\circ}$' % int(scale_bar_length / 3600)
        scale_bar_text_size = scale_bar_fontsize

        ax1.plot(
            [scale_bar_x_0, scale_bar_x_1], [scale_bar_y, scale_bar_y],
            linewidth=3,
            c=scale_bar_color,
            alpha=1.0)
        ax1.text(
            scale_bar_text_x,
            scale_bar_text_y,
            scale_bar_text,
            fontsize=scale_bar_text_size,
            horizontalalignment='center',
            color=scale_bar_color)
    if add_text is not None:
        text_x_0 = int(img_size_x*0.08)
        text_y_0 = int(img_size_y*text_y_offset)
        ax1.text(text_x_0, text_y_0, r'$\mathrm{'+add_text+'}$', fontsize=text_fontsize, color=text_color)

    # Put a color bar on the image
    if color_bar:
        ax_cbar = inset_axes(ax1,
                             width=color_bar_width,
                             height=color_bar_height,
                             loc=color_bar_loc)
        if ax is None:
            cbar = plt.colorbar(show, ax=ax1, cax=ax_cbar,
                                orientation='horizontal')
        else:
            cbar = plt.colorbar(show, ax=ax, cax=ax_cbar,
                                orientation='horizontal')

        cbar.ax.xaxis.set_tick_params(color=color_bar_color)
        cbar.ax.yaxis.set_tick_params(color=color_bar_color)
        cbar.outline.set_edgecolor(color_bar_color)
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)

    if ax is None:
        return fig
    return ax1

def _display_single(img,
                   pixel_scale=0.168,
                   physical_scale=None,
                   xsize=8,
                   ysize=8,
                   ax=None,
                   stretch='arcsinh',
                   scale='zscale',
                   scale_manual=None,
                   contrast=0.25,
                   no_negative=False,
                   lower_percentile=1.0,
                   upper_percentile=99.0,
                   cmap=IMG_CMAP,
                   scale_bar=True,
                   scale_bar_length=5.0,
                   scale_bar_fontsize=20,
                   scale_bar_y_offset=0.5,
                   scale_bar_color='w',
                   scale_bar_loc='left',
                   color_bar=False,
                   color_bar_loc=1,
                   color_bar_width='75%',
                   color_bar_height='5%',
                   color_bar_fontsize=18,
                   color_bar_color='w',
                   add_text=None,
                   text_fontsize=30,
                   text_y_offset=0.80,
                   text_color='w'):
    """Display single image. From `kungpao`.

    Parameters
    ----------
        img: np 2-D array for image

        xsize: int, default = 8
            Width of the image.

        ysize: int, default = 8
            Height of the image.

    """
    if ax is None:
        fig = plt.figure(figsize=(xsize, ysize))
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax

    # Stretch option
    if stretch.strip() == 'arcsinh':
        img_scale = np.arcsinh(img)
    elif stretch.strip() == 'log':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log(img)
    elif stretch.strip() == 'log10':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log10(img)
    elif stretch.strip() == 'linear':
        img_scale = img
    else:
        raise Exception("# Wrong stretch option.")

    # Scale option
    if scale.strip() == 'zscale':
        try:
            zmin, zmax = ZScaleInterval(contrast=contrast).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    elif scale.strip() == 'percentile':
        try:
            zmin, zmax = AsymmetricPercentileInterval(
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    else:
        zmin, zmax = np.nanmin(img_scale), np.nanmax(img_scale)
    
    if scale_manual is not None:
        assert len(scale_manual) == 2, '# length of manual scale must be two!'
        zmin, zmax = scale_manual

    show = ax1.imshow(img_scale, origin='lower', cmap=cmap,
                      vmin=zmin, vmax=zmax)

    # Hide ticks and tick labels
    ax1.tick_params(
        labelbottom=False,
        labelleft=False,
        axis=u'both',
        which=u'both',
        length=0)
    #ax1.axis('off')

    # Put scale bar on the image
    (img_size_x, img_size_y) = img.shape
    if physical_scale is not None:
        pixel_scale *= physical_scale
    if scale_bar:
        if scale_bar_loc == 'left':
            scale_bar_x_0 = int(img_size_x * 0.04)
            scale_bar_x_1 = int(img_size_x * 0.04 +
                                (scale_bar_length / pixel_scale))
        else:
            scale_bar_x_0 = int(img_size_x * 0.95 -
                                (scale_bar_length / pixel_scale))
            scale_bar_x_1 = int(img_size_x * 0.95)

        scale_bar_y = int(img_size_y * 0.10)
        scale_bar_text_x = (scale_bar_x_0 + scale_bar_x_1) / 2
        scale_bar_text_y = (scale_bar_y * scale_bar_y_offset)
        if physical_scale is not None:
            if scale_bar_length > 1000:
                scale_bar_text = r'$%d\ \mathrm{Mpc}$' % int(scale_bar_length / 1000)
            else:
                scale_bar_text = r'$%d\ \mathrm{kpc}$' % int(scale_bar_length)
        else:
            if scale_bar_length < 60:
                scale_bar_text = r'$%d^{\prime\prime}$' % int(scale_bar_length)
            elif 60 < scale_bar_length < 3600:
                scale_bar_text = r'$%d^{\prime}$' % int(scale_bar_length / 60)
            else: 
                scale_bar_text = r'$%d^{\circ}$' % int(scale_bar_length / 3600)
        scale_bar_text_size = scale_bar_fontsize

        ax1.plot(
            [scale_bar_x_0, scale_bar_x_1], [scale_bar_y, scale_bar_y],
            linewidth=3,
            c=scale_bar_color,
            alpha=1.0)
        ax1.text(
            scale_bar_text_x,
            scale_bar_text_y,
            scale_bar_text,
            fontsize=scale_bar_text_size,
            horizontalalignment='center',
            color=scale_bar_color)
    if add_text is not None:
        text_x_0 = int(img_size_x*0.08)
        text_y_0 = int(img_size_y*text_y_offset)
        ax1.text(text_x_0, text_y_0, r'$\mathrm{'+add_text+'}$', fontsize=text_fontsize, color=text_color)

    # Put a color bar on the image
    if color_bar:
        ax_cbar = inset_axes(ax1,
                             width=color_bar_width,
                             height=color_bar_height,
                             loc=color_bar_loc)
        if ax is None:
            cbar = plt.colorbar(show, ax=ax1, cax=ax_cbar,
                                orientation='horizontal')
        else:
            cbar = plt.colorbar(show, ax=ax, cax=ax_cbar,
                                orientation='horizontal')

        cbar.ax.xaxis.set_tick_params(color=color_bar_color)
        cbar.ax.yaxis.set_tick_params(color=color_bar_color)
        cbar.outline.set_edgecolor(color_bar_color)
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)

    if ax is None:
        return fig, zmin, zmax
    return ax1, zmin, zmax

def display_multiple(data_array, text=None, ax=None, **kwargs):
    if ax is None:
        fig, axes = plt.subplots(1, len(data_array), figsize=(len(data_array) * 4, 8))
    else:
        axes = ax

    if text is None:
        _, zmin, zmax = _display_single(data_array[0], ax=axes[0], **kwargs)
    else:
        _, zmin, zmax = _display_single(data_array[0], add_text=text[0], ax=axes[0], **kwargs)
    for i in range(1, len(data_array)):
        if text is None:
            _display_single(data_array[i], ax=axes[i], scale_manual=[zmin, zmax], scale_bar=False, **kwargs)
        else:
            _display_single(data_array[i], add_text=text[i], ax=axes[i], scale_manual=[zmin, zmax], scale_bar=False, **kwargs)

    plt.subplots_adjust(wspace=0.0)
    if ax is None:
        return fig
    else:
        return axes


def draw_circles(img, catalog, colnames=['x', 'y'], header=None, ax=None, circle_size=30, 
                 pixel_scale=0.168, color='r', **kwargs):
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(left=0.0, right=1.0, 
                            bottom=0.0, top=1.0,
                            wspace=0.00, hspace=0.00)
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace=0.0, hspace=0.00)
        ax1 = fig.add_subplot(gs[0])
    else:
        ax1 = ax

    #ax1.yaxis.set_major_formatter(NullFormatter())
    #ax1.xaxis.set_major_formatter(NullFormatter())
    ax1.axis('off')
    
    from matplotlib.patches import Ellipse, Rectangle
    if np.any([item.lower() == 'ra' for item in colnames]): 
        if header is None:
            raise ValueError('# Header containing WCS must be provided to convert sky coordinates into image coordinates.')
            return
        else:
            w = wcs.WCS(header)
            x, y = w.wcs_world2pix(Table(catalog)[colnames[0]].data.data, 
                                   Table(catalog)[colnames[1]].data.data, 0)
    else:
        x, y = catalog[colnames[0]], catalog[colnames[1]]
    display_single(img, ax=ax1, pixel_scale=pixel_scale, **kwargs)
    for i in range(len(catalog)):
        e = Ellipse(xy=(x[i], y[i]),
                        height=circle_size,
                        width=circle_size,
                        angle=0)
        e.set_facecolor('none')
        e.set_edgecolor(color)
        e.set_alpha(0.7)
        e.set_linewidth(1.3)
        ax1.add_artist(e)
    if ax is not None:
        return ax

def draw_rectangles(img, catalog, colnames=['x', 'y'], header=None, ax=None, rectangle_size=[30, 30], 
                    pixel_scale=0.168, color='r', **kwargs):
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(left=0.0, right=1.0, 
                            bottom=0.0, top=1.0,
                            wspace=0.00, hspace=0.00)
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace=0.0, hspace=0.00)
        ax1 = fig.add_subplot(gs[0])
    else:
        ax1 = ax

    #ax1.yaxis.set_major_formatter(NullFormatter())
    #ax1.xaxis.set_major_formatter(NullFormatter())
    #ax1.axis('off')
    
    from matplotlib.patches import Rectangle
    if np.any([item.lower() == 'ra' for item in colnames]): 
        if header is None:
            raise ValueError('# Header containing WCS must be provided to convert sky coordinates into image coordinates.')
            return
        else:
            w = wcs.WCS(header)
            x, y = w.wcs_world2pix(Table(catalog)[colnames[0]].data.data, 
                                   Table(catalog)[colnames[1]].data.data, 0)
    else:
        x, y = catalog[colnames[0]], catalog[colnames[1]]
    display_single(img, ax=ax1, pixel_scale=pixel_scale, **kwargs)
    for i in range(len(catalog)):
        e = Rectangle(xy=(x[i] - rectangle_size[0] // 2, 
                          y[i] - rectangle_size[1] // 2),
                        height=rectangle_size[0],
                        width=rectangle_size[1],
                        angle=0)
        e.set_facecolor('none')
        e.set_edgecolor(color)
        e.set_alpha(0.7)
        e.set_linewidth(1.3)
        ax1.add_artist(e)
    if ax is not None:
        return ax


# You can plot 1-D SBP using this, without plotting the PA and eccentricity.
def psf_sbp(ell_fix, pixel_scale=0.168, ax=None, x_min=0.0, x_max=5.0, alpha=1, 
    show_dots=False, show_grid=False, vertical_line=None, 
    linecolor='firebrick', linestyle='-', linewidth=3, labelsize=25, 
    ticksize=30, label='SBP', labelloc='lower left'):

    """Display the 1-D profiles, without showing PA and ellipticity.
    
    Parameters:
    -----------
    ell_fix: astropy Table or numpy table, should be the output of ELLIPSE.
    pixel_scale: float, pixel scale in arcsec/pixel.
    zeropoint: float, zeropoint of the photometry system.
    ax: matplotlib axes class.
    offset: float.
    x_min, x_max: float, in ^{1/4} scale.
    alpha: float, transparency.
    physical_unit: boolean. If true, the figure will be shown in physical scale.
    show_dots: boolean. If true, it will show all the data points.
    show_grid: boolean. If true, it will show a grid.
    vertical_line: list of floats, positions of vertical lines. Maximum length is three.
    linecolor, linestyle: string. Color and style of SBP.
    label: string.

    Returns:
    --------
    ax: matplotlib axes class.

    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(left=0.0, right=1.0, 
                            bottom=0.0, top=1.0,
                            wspace=0.00, hspace=0.00)

        ax1 = fig.add_axes([0.08, 0.07, 0.85, 0.88])
        ax1.tick_params(direction='in')
    else:
        ax1 = ax
        ax1.tick_params(direction='in')

    # 1-D profile

    x = ell_fix['sma'] * pixel_scale
    y = np.log10((ell_fix['intens']) / (pixel_scale)**2)
    y_upper = np.log10((ell_fix['intens'] + ell_fix['int_err']) / (pixel_scale) ** 2)
    y_lower = np.log10((ell_fix['intens'] - ell_fix['int_err']) / (pixel_scale) ** 2)
    upper_yerr = y_lower - y
    lower_yerr = y - y_upper
    asymmetric_error = [lower_yerr, upper_yerr]
    xlabel = r'$R/\mathrm{arcsec}$'
    #ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'

    if show_grid:
        ax1.grid(linestyle='--', alpha=0.4, linewidth=2)
    if show_dots:
        ax1.errorbar(x, y,
                 yerr=asymmetric_error,
                 color='k', alpha=0.2, fmt='o', 
                 capsize=4, capthick=1, elinewidth=1)

    if label is not None:
        ax1.plot(x, y, color=linecolor, linewidth=linewidth, linestyle=linestyle,
             label=r'$\mathrm{' + label + '}$', alpha=alpha)
        leg = ax1.legend(fontsize=labelsize, frameon=False, loc=labelloc)
        for l in leg.legendHandles:
            l.set_alpha(1)
    else:
        ax1.plot(x, y, color=linecolor, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
    ax1.fill_between(x, y_upper, y_lower, color=linecolor, alpha=0.3*alpha)
    
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)

    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(xlabel, fontsize=ticksize)
    ax1.set_yticklabels([])
    #yticks = ax1.get_yticks()
    #ax1.set_yticklabels([r'$10^{' + str(int(u)) + '}$' for u in yticks])
    ax1.invert_yaxis()

    # Vertical line
    if vertical_line is not None:
        if len(vertical_line) > 3:
            raise ValueError('Maximum length of vertical_line is 3.') 
        ylim = ax1.get_ylim()
        style_list = ['-', '--', '-.']
        for k, pos in enumerate(vertical_line):
            ax1.axvline(x=pos, ymin=0, ymax=1,
                        color='gray', linestyle=style_list[k], linewidth=3, alpha=0.75)
        plt.ylim(ylim)

    # Return
    if ax is None:
        return fig
    return ax1
