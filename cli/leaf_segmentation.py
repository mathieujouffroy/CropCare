#from typing import final
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cli_utils import plot_multiple_img
from image_preprocessing import *


def distance_transform_fb(rgb_img, hsv_mask, fill_sbg=True, verbose=False):
    """
    Distance transform from foreground to backgroundotsu= applied on the hsv mask.

    Args:
        rgb_img (numpy.array):      RGB image
        hsv_mask (numpy.array):     Hue, Saturation, Value mask

    Returns:
        mask (numpy.array):         mask on leaves with distance transform
        markers (numpy.array):      mask with connected components
    """

    # Noise Removal with morph transform
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=8)

    # fill holes in the foreground mask
    if fill_sbg == True:
        cnts = cv2.findContours(
            sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        sure_bg = cv2.fillPoly(sure_bg, cnts, (255, 255, 255))

    # distance transform from background to foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    dist_transform = cv2.normalize(
        dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.11*dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    nb_connected, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    tmp_img = copy.deepcopy(rgb_img)
    markers = cv2.watershed(tmp_img, markers)

    # set sur fg to white
    markers[markers > 1] = 255
    markers = np.uint8(markers)
    markers = cv2.morphologyEx(markers, cv2.MORPH_CLOSE, kernel, iterations=1)

    bg_mask = copy.deepcopy(rgb_img)
    # set background to black
    bg_mask[markers <= 1] = [0, 0, 0]

    if verbose:
        imgs = [hsv_mask, opening, sure_bg, dist_transform,
                sure_fg, unknown, markers, bg_mask]
        titles = ['hsv_mask', 'opening', 'sure_bg', 'dist_transform',
                  'sure_fg', 'bg - fg', 'markers', 'with mask']
        plot_multiple_img(imgs, True, titles=titles)

    return bg_mask


def color_mask(rgb_img, ls1, ls2, type=1, verbose=False):
    """
    HSV Color mask applied on the rgb image.
    Args:
        rgb_img (numpy.array):     RGB image
        ls1 (int):                 lower saturation on yellow-green-blue
        ls2 (int):                 lower saturation on red-brown-orange
        type (int):                type of mask
    Returns:
        final_mask (numpy.array):   mask on leaves
        result (numpy.array):       image with a mask on yellow-green-blue
        disease_result (numpy.array): image with a mask on red-brown-orange
    """
    # Remove noise with blur and transform to HSV
    g_blurred = cv2.GaussianBlur(rgb_img, (5, 5), 0)
    #g_blurred = cv2.GaussianBlur(rgb_img, (3, 3), 0)
    hsv = cv2.cvtColor(g_blurred, cv2.COLOR_RGB2HSV)

    # define kernel for morphology transformations
    k_size = (5, 5)  # (3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)

    # mask on yellow-green-blue
    lower_green = np.array([20, ls1, 25])
    upper_green = np.array([103, 255, 255])
    healthy_mask = cv2.inRange(hsv, lower_green, upper_green)
    healthy_mask = cv2.morphologyEx(healthy_mask, cv2.MORPH_CLOSE, kernel)

    if type == 1:
        # mask on red
        lower_red = np.array([0, 75, 28])
        upper_red = np.array([15, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        # mask on orange-yellow
        lower_yellow = np.array([15, 50, 28])
        upper_yellow = np.array([30, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # mask on upper red
        upper_red_brown = cv2.inRange(hsv, (160, 70, 30), (180, 200, 200))
        disease_mask = orange_mask | red_mask
        disease_mask = cv2.morphologyEx(
            disease_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # combine the masks
        disease_mask = disease_mask | upper_red_brown
        final_mask = healthy_mask | disease_mask

    else:
        # mask on red-brown-orange
        lower_brown = np.array([0, ls2, 25])
        upper_brown = np.array([30, 255, 255])
        disease_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)
        upper_red_brown = cv2.inRange(hsv, (160, 70, 30), (180, 200, 200))
        # combine the masks
        disease_mask = disease_mask | upper_red_brown
        final_mask = healthy_mask | disease_mask

    # apply each mask on original RGB image
    result = cv2.bitwise_and(rgb_img, rgb_img, mask=healthy_mask)
    disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)

    # Mask on white on green leaves
    white_mask = cv2.inRange(hsv, (25, 10, 190), (100, 100, 255))
    white_res = cv2.bitwise_and(rgb_img, rgb_img, mask=white_mask)

    final_mask = final_mask | white_mask
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    if verbose:
        plot_multiple_img(imgs=[result, white_res, disease_result, ],
                          gray=True, titles=['green-ish', 'brown-ish', 'white-ish'])

    return final_mask, result, disease_result


def back_segmentation(rgb_img, white=True, dist=False, lightness=False, contrast=False, cast=False, verbose=False):
    """
    Image segmentation using background subtraction.

    Args:
        rgb_img (numpy.array):      RGB image
        dist (bool):                option to use distance transformation
        white (bool):               option to remove white pixels from mask
        lightness (bool):           option to adjust lightness in image
        contrast (bool):            option to adjust contrast in image
    Returns:
        no_back_img (numpy.array):  RGB image with the background removed
    """

    new_img = rgb_img.copy()

    if cast: new_img = color_cast_removal(new_img)
    if contrast: new_img = adjust_contrast(new_img)
    if lightness: new_img = adjust_lightness(new_img)
    #test = automatic_brightness_and_contrast(rgb_img, verbose=True)
    #new_img = adjust_gamma(new_img)


    final_mask, result, disease_result = color_mask(
        new_img, ls1=17, ls2=60)

    # if image is not of white majority -> remove whites on leaves
    if white:
        final_mask = remove_whites(rgb_img, final_mask)
    if dist:
        dist_transf = distance_transform_fb(rgb_img, final_mask)

    final_mask, no_back_img = fill_object(rgb_img, final_mask)

    if verbose:
        imgs = [rgb_img, new_img, result, disease_result, final_mask, no_back_img]
        titles = ['rgb img', 'new_img', 'mask on blue-green-yellow',
                  'mask on red-brown', 'HSV_mask', 'back segm img']
        if dist:
            imgs.append(dist_transf)
            titles.append('dist transf')
        plot_multiple_img(imgs=imgs, gray=True, titles=titles)

    if dist:
        no_back_img = dist_transf
    return no_back_img


def remove_background(rgb_img, p_type, dist=False, morphs=False, adapt_th=False, verbose=False):
    """
    Remove background from RGB image.

    Args:
        rgb_img (numpy.array):   RGB image
        p_type (int):            type of image preprocessing
        dist(bool):              option to use distance transformations
        morphs (bool):           option to use morphological operations
        adapt_th (bool):         option to use adaptive thresholding
    Returns:
        no_back_img (numpy.array): RGB image with the background removed
    """
    image_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    vb = verbose

    if morphs:
        morph_transform(image_gray)
    if adapt_th:
        adaptive_thresh_and_canny(image_gray)

    if p_type == 0:
        #print("----- ORIGINAL -----")
        no_back_img = back_segmentation(rgb_img,  dist=dist, verbose=vb)
    elif p_type == 1:
        #print("----- LIGHTNESS ADJUSTED -----")
        no_back_img = back_segmentation(
            rgb_img, dist=dist, lightness=True, verbose=vb)
    elif p_type == 2:
        #print("----- CONTRASTED -----")
        no_back_img = back_segmentation(
            rgb_img, dist=dist, contrast=True, verbose=vb)
    elif p_type == 3:
        #print("----- CONTRAST & LIGHTNESS ADJUSTED -----")
        no_back_img = back_segmentation(
            rgb_img, dist=dist, lightness=True, contrast=True, verbose=vb)
    return no_back_img
