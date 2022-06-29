import cv2
import numpy as np
import matplotlib.pyplot as plt
from cli_utils import plot_multiple_img
import skimage


def adaptive_thresh_and_canny(image_gray):
    """
    Uses a blurring function, adaptive thresholding and dilation to expose
    the main features of an image.
    Args:
        image_gray (numpy.ndarray):    grayscale image to process
    """
    blur = cv2.medianBlur(image_gray.copy(), 5)
    thresh = [
        cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2),
        cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2),

        cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3),

        cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 4),
        cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4),

        cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2),
    ]

    canny = cv2.Canny(image_gray, 30, 100)
    k_size = (3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)
    final = []
    for th in thresh:
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        final.append(th)
    final.append(canny)

    titles = ['AdapMeanTh 15-2', 'AdapGaussTh 15-2', 'AdapMeanTh 15-3',
              'AdapMeanTh 21-4', 'AdapGaussTh 21-4', 'AdapMeanTh 21-2',
              'canny edges --- [30-150]']
    plot_multiple_img(final, True, titles)


def morph_transform(image_gray):
    """
    Applies morphological transformations to a gray_scale image. These
    operations “probe” an image with a structuring element which defines the
    neighborhood to be examined around each pixel.
    Args:
        image_gray (numpy.ndarray):    grayscale image to process
    """
    k_size = (5, 5)
    eroded = cv2.erode(image_gray.copy(), None, iterations=1)
    dilated = cv2.dilate(image_gray.copy(), None, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)
    opening = cv2.morphologyEx(image_gray.copy(), cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image_gray.copy(), cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(image_gray.copy(), cv2.MORPH_GRADIENT, kernel)
    morphs = [eroded, dilated, opening, closing, gradient]
    titles = ["eroded with 1 iter", 'dilated with 1 iter',
              f"opening on {k_size} - morph ellipse",
              f"closing on {k_size} - morph ellipse",
              f"gradient on {k_size} - morph ellipse"]
    plot_multiple_img(morphs, True, titles)


def automatic_brightness_and_contrast(rgb_img, clip_hist_percent=2, verbose=False):
    """
    Automatic brightness and contrast optimization with histogram clipping.
    Args:
        rgb_img (numpy.ndarray):    RGB image to adjust
        clip_hist_percent (int):    specify the histogram percentile to clip at
        verbose (bool):             option to display images
    Returns:
        new_img (numpy.ndarray):    image with brightness and contrast adjusted
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    new_img = cv2.convertScaleAbs(rgb_img, alpha=alpha, beta=beta)
    if verbose:
        plot_multiple_img([rgb_img, new_img], True,
                          titles=['image', 'new image with auto bright+contrast'])
    return new_img


def adjust_lightness(rgb_img, verbose=False):
    """
    Adjust the lightness of an RGB image.
    Args:
        rgb_img (numpy.ndarray):    RGB image to adjust
        verbose (bool):             option to display the images
    Returns:
        final (numpy.ndarray):      image with adjusted lightness
    """
    # Convert image to LAB color space
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Define the lightness boundaries
    upper_lim = np.max(l) - 30
    lower_lim = np.min(l) + 30

    # Adjust the lightness
    new_l = np.where(l <= lower_lim, lower_lim, l)
    new_l = np.where(l >= upper_lim, upper_lim, new_l)
    new_lab = cv2.merge((new_l, a, b))

    final = cv2.cvtColor(new_lab, cv2.COLOR_LAB2RGB)
    if verbose:
        plot_multiple_img([rgb_img, l, new_l, lab, new_lab, final], gray=True,
                          titles=['rgb', 'l', 'new_l', 'lab', 'new_lab', 'final'])
    return final


def adjust_contrast(rgb_img, verbose=False):
    """
    Adjust the contrast of an RGB image using the Contrast Limited Adaptive
    Histogram Equalization (CLAHE) method.
    Args:
        rgb_img (numpy.ndarray):    RGB image to adjust
        verbose (bool):             option to display images
    Returns:
        new_img (numpy.ndarray):      image with adjusted contrast
    """
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    c_l = clahe.apply(l)

    new_lab = cv2.merge((c_l, a, b))
    new_img = cv2.cvtColor(new_lab, cv2.COLOR_LAB2RGB)
    if verbose:
        imgs = [rgb_img, l, c_l, lab, new_lab, new_img]
        titles = ['original', 'l channel', 'l CLAHE, lim=3, size=(8,8)',
                  'lab', 'new lab', 'new_img with CLAHE applied']
        plot_multiple_img(imgs, gray=True, titles=titles)
    return new_img


def remove_whites(image, mask):
    """
    Remove pixels resembling white from mask as background
    Args:
        image: RGB image
        mask:  mask to be cleaned
    Returns:
        mask: mask with white pixels removed
    """
    # setup the white remover to process logical_and in place
    white_remover = np.full((image.shape[0], image.shape[1]), 0)
    white_remover[image[:, :, 0] >= 250] = 255
    white_remover[image[:, :, 1] >= 200] = 255
    white_remover[image[:, :, 2] >= 250] = 255
    # remove whites from mask
    mask[white_remover] = 255
    return mask


def remove_blacks(image, marker):
    """
    Remove pixels resembling black from marker as background
    Args:
        image:
        marker: to be overloaded with black pixels to be removed
    Returns:
        nothing
    """
    # setup the black remover to process logical_and in place
    black_remover = np.full((image.shape[0], image.shape[1]), True)
    black_remover[image[:, :, 0] >= 30] = False  # blue channel
    black_remover[image[:, :, 1] >= 30] = False  # green channel
    black_remover[image[:, :, 2] >= 30] = False  # red channel
    # remove blacks from marker
    marker[black_remover] = False
    plt.imshow(black_remover)
    plt.show()


def fill_object(rgb_img, final_mask):
    """
    Fills the object in the mask with white pixels.
    Args:
        rgb_img (numpy.ndarray):    RGB image to process
        final_mask (numpy.ndarray): mask to extract the object to fill
    Returns:
        final_mask (numpy.ndarray): mask with object filled with white pixels
        no_back_img (numpy.ndarray): RGB image with final_mask applied
    """
    cnts = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # fill polygon with white pixels given end points (cnts)
    final_mask = cv2.fillPoly(final_mask, cnts, (255, 255, 255))
    no_back_img = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    return final_mask, no_back_img


def color_cast_removal(rgb_img, verbose=False):
    """
    Removes unwanted tint of colors.
    Args:
        rgb_img (numpy.ndarray):    RGB image to process
        verbose (bool):             option to display images
    """
    hsv_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    # separate channels
    h, s, v = cv2.split(hsv_image)
    # reverse the hue channel by 180 deg out of 360, so in python add 90 and modulo 180
    h_new = (h + 90) % 180
    # combine new hue with old sat and value
    hsv_new = cv2.merge([h_new, s, v])
    # convert back to BGR
    rgb_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB)
    # Get the average color of rgb_new
    ave_color = cv2.mean(rgb_new)[0:3]
    print(ave_color)

    # create a new image with the average color
    color_img = np.full_like(rgb_img, ave_color)
    # make a 50-50 blend of img and color_img
    blend = cv2.addWeighted(rgb_img, 0.5, color_img, 0.5, 0.0)
    # stretch dynamic range
    result = skimage.exposure.rescale_intensity(
        blend, in_range='image', out_range=(0, 255)).astype(np.uint8)

    if verbose:
        plot_multiple_img(imgs=[rgb_img, hsv_image, hsv_new, result], gray=True,
                          titles=['rgb_img', 'hsv_image', 'hsv_new', 'result'])
    return result
