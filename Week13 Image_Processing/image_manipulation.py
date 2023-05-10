
# Import the necessary packages
import numpy as np
from matplotlib import pyplot as plt

from skimage import io
from skimage import img_as_ubyte, img_as_uint, img_as_float64

from skimage import data,exposure
from skimage.transform import rotate

from sklearn.cluster import KMeans
import argparse

def plot_channel_intensities(image):

    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]

    mask = ( red + green + blue ) > 0

    fig, ax_hist = plt.subplots(1, 1, figsize=(6, 5))

    # histogram of each
    ax_hist.hist(red[mask].ravel(), bins=256, histtype='step', color='red')
    ax_hist.hist(green[mask].ravel(), bins=256, histtype='step', color='green')
    ax_hist.hist(blue[mask].ravel(), bins=256, histtype='step', color='blue')

    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    ax_hist.set_ylabel('Pixel Count')
    ax_hist.set_xlabel('Pixel intensity')

def rescale_intensities(image,flow,fhigh):

    # Scale the intensities and replot the histogram

    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]

    mask = ( red + green + blue ) > 0

    # Pick percentile overwhich to rescale intensities
    flow = 2
    fhigh = 98

    plow, phigh = np.percentile(red[mask], (flow, fhigh))
    ls_red_rs= exposure.rescale_intensity(red, in_range=(plow, phigh))

    plow, phigh = np.percentile(green[mask], (flow, fhigh))
    ls_green_rs= exposure.rescale_intensity(green, in_range=(plow, phigh))

    plow, phigh = np.percentile(blue[mask], (flow, fhigh))
    ls_blue_rs= exposure.rescale_intensity(blue, in_range=(plow, phigh))

    # Define empty image
    nx,ny = ls_red_rs.shape
    ls_rgb_stretched = np.zeros([nx,ny,3],dtype=np.float64)

    # set all 3 channels of the image
    ls_rgb_stretched[:,:,0] = ls_red_rs
    ls_rgb_stretched[:,:,1] = ls_green_rs
    ls_rgb_stretched[:,:,2] = ls_blue_rs

    return  ls_rgb_stretched


def discretize_image(N,image):

    # reshape the image to be a list of pixels
    pixels = image.reshape((image.shape[0] * image.shape[1], 3))
    pixels.shape

    # cluster the pixel intensities
    clt = KMeans(n_clusters = N)
    clt.fit(pixels)

    # Round color values and convert to uint8
    palette = np.round(clt.cluster_centers_).astype('uint8')

    # build labeled image
    image_label = clt.labels_.reshape((image.shape[0], image.shape[1]))

    # replace the labels with the pixels
    pixels = np.array([ palette[l] for l in list(clt.labels_)])
    image_disc = pixels.reshape((image.shape[0], image.shape[1], 3))

    return image_disc, image_label, palette


def bar_plot_with_colors(hist,colors):

    # Matplotlib takes an RGB *fraction* as input for colors
    cm = [tuple(1.*np.array(c)/255.) for c in colors] # Matplotlib colormap takes a fraction

    plt.figure(figsize=(10, 3))
    plt.title('Fractional area of color clusters')
    plt.subplot(121)
    plt.bar(range(len(hist)),hist,color=cm)
