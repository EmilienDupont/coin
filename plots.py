# Based on https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/utils/plot/__main__.py
import imageio
import json5 as json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import cm
from pathlib import Path


ours = 'COIN'

# Ensure consistent coloring across plots
name_to_color = {
    ours: mcolors.TABLEAU_COLORS['tab:blue'],
    'BMS': mcolors.TABLEAU_COLORS['tab:orange'],
    'MBT': mcolors.TABLEAU_COLORS['tab:green'],
    'CST': mcolors.TABLEAU_COLORS['tab:red'],
    'JPEG': mcolors.TABLEAU_COLORS['tab:purple'],
    'JPEG2000': mcolors.TABLEAU_COLORS['tab:brown'],
    'BPG': mcolors.TABLEAU_COLORS['tab:pink'],
    'VTM': mcolors.TABLEAU_COLORS['tab:gray'],
}

# Setup colormap for residuals plot
viridis = cm.get_cmap('viridis', 100)


def parse_json_file(filepath, metric='psnr'):
    """Parses a json result file.

    Args:
        filepath (string): Path to results json file.
        metric (string): Metric to use for plot.
    """
    filepath = Path(filepath)
    name = filepath.name.split('.')[0]
    with filepath.open('r') as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError as err:
            print(f'Error reading file {filepath}')
            raise err

    if 'results' not in data or 'bpp' not in data['results']:
        raise ValueError(f'Invalid file {filepath}')

    if metric not in data['results']:
        raise ValueError(
            f'Error: metric {metric} not available.'
            f' Available metrics: {", ".join(data["results"].keys())}'
        )

    if metric == 'ms-ssim':
        # Convert to db
        values = np.array(data['results'][metric])
        data['results'][metric] = -10 * np.log10(1 - values)

    return {
        'name': data.get('name', name),
        'xs': data['results']['bpp'],
        'ys': data['results'][metric],
    }


def rate_distortion(scatters, title=None, ylabel='PSNR [dB]', output_file=None,
                    limits=None, show=False, figsize=None):
    """Creates a rate distortion plot based on scatters.

    Args:
        scatters (list of dicts): List of data to plot for each model.
        title (string):
        ylabel (string):
        output_file (string): If not None, save plot at output_file.
        limits (tuple of ints):
        show (bool): If True shows plot.
        figsize (tuple of ints):
    """
    if figsize is None:
        figsize = (7, 4)
    fig, ax = plt.subplots(figsize=figsize)
    for sc in scatters:
        if sc['name'] == ours:
            linewidth = 2.5
            markersize = 10
        else:
            linewidth = 1
            markersize = 6

        if sc['name'] in [ours, 'BMS', 'MBT', 'CST']:
            pattern = '.-'  # Learned algorithms
        else:
            pattern = '.--'  # Non learned algorithms
        ax.plot(sc['xs'], sc['ys'], pattern, label=sc['name'],
                c=name_to_color[sc['name']], linewidth=linewidth,
                markersize=markersize)

    ax.set_xlabel('Bit-rate [bpp]')
    ax.set_ylabel(ylabel)
    ax.grid()
    if limits is not None:
        ax.axis(limits)
    ax.legend(loc='lower right')

    if title:
        ax.title.set_text(title)

    if show:
        plt.show()

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()


def plot_rate_distortion(filepaths=['results.json',
                                    'baselines/compressai-bmshj2018-hyperprior.json',
                                    'baselines/compressai-mbt2018.json',
                                    'baselines/compressai-cheng2020-anchor.json',
                                    'baselines/jpeg.json', 'baselines/jpeg2000.json',
                                    'baselines/bpg_444_x265_ycbcr.json',
                                    'baselines/vtm.json'],
                         output_file=None, limits=None):
    """Creates rate distortion plot based on all results json files.

    Args:
        filepaths (list of string): List of paths to result json files.
        output_file (string): Path to save image.
        limits (tuple of float): Limits of plot.
    """
    # Read data
    scatters = []
    for f in filepaths:
        rv = parse_json_file(f, 'psnr')
        scatters.append(rv)
    # Create plot
    rate_distortion(scatters, output_file=output_file, limits=limits)


def plot_model_size(output_file=None, show=False):
    """Plots histogram of model sizes.

    Args:
        output_file (string): If not None, save plot at output_file.
        show (bool): If True shows plot.

    Notes:
        Data for all baselines was computed using the compressAI library
        https://github.com/InterDigitalInc/CompressAI
    """
    model_names = ['COIN', 'BMS', 'MBT', 'CST']
    model_sizes = [14.7455, 10135.868, 24764.604, 31834.464]  # in kB

    plt.grid(zorder=0, which="both", axis="y")  # Ensure grid is at the back

    barplot = plt.bar(model_names, model_sizes, log=True, zorder=10)
    for i in range(len(model_names)):
        barplot[i].set_color(name_to_color[model_names[i]])
    plt.ylabel("Model size [kB]")

    fig = plt.gcf()
    fig.set_size_inches(3, 4)

    if show:
        plt.show()

    if output_file:
        plt.savefig(output_file, format='png', dpi=400, bbox_inches='tight')
        plt.clf()
        plt.close()


def plot_residuals(path_original='kodak-dataset/kodim15.png',
                   path_coin='imgs/kodim15_coin_bpp_03.png',
                   path_jpeg='imgs/kodim15_jpeg_bpp_03.jpg',
                   output_file=None, show=False, max_residual=0.3,
                   title_fontsize=6):
    """Creates a plot comparing compression with COIN and JPEG both in terms of
    the compressed image and the residual between the compressed and original
    image.


    Args:
        path_original (string): Path to original image.
        path_coin (string): Path to image compressed with COIN.
        path_jpeg (string): Path to image compressed with JPEG.
        output_file (string): If not None, save plot at output_file.
        show (bool): If True shows plot.
        max_residual (float): Value between 0 and 1 to use for maximum residual
            on color scale. Usually set to a low value so residuals are clearer
            on plot.
    """
    # Load images and compute residuals
    img_original = imageio.imread(path_original) / 255.
    img_coin = imageio.imread(path_coin) / 255.
    img_jpeg = imageio.imread(path_jpeg) / 255.
    residual_coin = viridis(np.abs(img_coin - img_original).mean(axis=-1) / max_residual)[:, :, :3]
    residual_jpeg = viridis(np.abs(img_jpeg - img_original).mean(axis=-1) / max_residual)[:, :, :3]

    # Create plot
    plt.subplot(2, 3, 1)
    plt.imshow(img_original)
    plt.axis('off')
    plt.gca().set_title('Original', fontsize=title_fontsize)

    plt.subplot(2, 3, 2)
    plt.imshow(img_coin)
    plt.axis('off')
    plt.gca().set_title('COIN', fontsize=title_fontsize)

    plt.subplot(2, 3, 3)
    plt.imshow(residual_coin)
    plt.axis('off')
    plt.gca().set_title('COIN Residual', fontsize=title_fontsize)

    plt.subplot(2, 3, 5)
    plt.imshow(img_jpeg)
    plt.axis('off')
    plt.gca().set_title('JPEG', fontsize=title_fontsize)

    plt.subplot(2, 3, 6)
    plt.imshow(residual_jpeg)
    plt.axis('off')
    plt.gca().set_title('JPEG Residual', fontsize=title_fontsize)

    plt.subplots_adjust(wspace=0.1, hspace=0)

    if show:
        plt.show()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()


if __name__ == '__main__':
    plot_rate_distortion(output_file='rate_distortion.png',
                         limits=(0, 1, 22, 38))
    plot_model_size(output_file='model_sizes.png')
    plot_residuals(output_file='residuals_kodim15_bpp_03.png')
    plot_residuals(output_file='residuals_kodim15_bpp_015.png',
                   path_coin='imgs/kodim15_coin_bpp_015.png',
                   path_jpeg='imgs/kodim15_jpeg_bpp_015.jpg')
