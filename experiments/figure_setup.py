import math
import matplotlib.pyplot as plt
import os
import subprocess
import tempfile


def get_fig_size(fig_width_cm, fig_height_cm=None):
    """Convert dimensions in centimeters to inches.
    If no height is given, it is computed using the golden ratio.
    """
    if not fig_height_cm:
        golden_ratio = (1 + math.sqrt(5))/2
        fig_height_cm = fig_width_cm / golden_ratio

    size_cm = (fig_width_cm, fig_height_cm)
    return map(lambda x: x/2.54, size_cm)


"""
The following functions can be used by scripts to get the sizes of
the various elements of the figures.
"""


def label_size():
    """Size of axis labels
    """
    return 8


def font_size():
    """Size of all texts shown in plots
    """
    return 8


def ticks_size():
    """Size of axes' ticks
    """
    return 6


def axis_lw():
    """Line width of the axes
    """
    return 0.6


def plot_lw():
    """Line width of the plotted curves
    """
    return 1.0


def figure_setup():
    """Set all the sizes to the correct values and use
    tex fonts for all texts.
    """

    params = {'text.usetex': False,
              'figure.dpi': 150,
              'font.size': font_size(),
              'font.sans-serif': ['helvetica', 'Arial'],
              'font.serif': ['Times', 'NimbusRomNo9L-Reg', 'TeX Gyre Termes', 'Times New Roman'],
              'font.monospace': [],
              'lines.linewidth': plot_lw(),
              'axes.labelsize': label_size(),
              'axes.titlesize': font_size(),
              'axes.linewidth': axis_lw(),
              'legend.fontsize': font_size(),
              'xtick.labelsize': ticks_size(),
              'ytick.labelsize': ticks_size(),
              'font.family': 'sans-serif',
              'xtick.bottom': False,
              'xtick.top': False,
              'ytick.left': False, 
              'ytick.right': False,
              'xtick.major.pad': -1,
              'ytick.major.pad': -2,
              'xtick.minor.visible': False,
              'ytick.minor.visible': False,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'axes.labelpad': 0,
              'axes.titlepad': 3,
              'axes.unicode_minus': False,
              'pdf.fonttype': 42,
              'ps.fonttype': 42}
              #'figure.constrained_layout.use': True}
    plt.rcParams.update(params)


def save_fig(fig, file_name, fmt=None, dpi=150, tight=False):
    """Save a Matplotlib figure as EPS/PNG/PDF to the given path and trim it.
    """

    if not fmt:
        fmt = file_name.strip().split('.')[-1]

    if fmt not in ['eps', 'png', 'pdf']:
        raise ValueError('unsupported format: %s' % (fmt,))

    extension = '.%s' % (fmt,)
    if not file_name.endswith(extension):
        file_name += extension

    file_name = os.path.abspath(file_name)
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_name = tmp_file.name + extension

    # save figure
    if tight:
        #fig.savefig(tmp_name, dpi=dpi, bbox_inches='tight')
        fig.savefig(file_name, dpi=dpi, bbox_inches='tight', pad_inches=0)
    else:
        fig.savefig(file_name, dpi=dpi)

    # trim it
    if fmt == 'eps':
        subprocess.call('epstool --bbox --copy %s %s' %
                        (tmp_name, file_name), shell=True)
    elif fmt == 'png':
        subprocess.call('convert %s -trim %s' %
                        (tmp_name, file_name), shell=True)
    #elif fmt == 'pdf':
    #    subprocess.call('pdfcrop %s %s' % (tmp_name, file_name), shell=True)
