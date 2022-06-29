import cv2
import numpy as np
import plantcv as pcv

def mask_h_pixels(rgb_img):
  hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
  value = 30
  lim = 255 - value
  #healthy_mask = cv2.inRange(hsv, lower_green, upper_green)
  hsv[..., 2] = np.where(hsv[..., 2] >= lim, lim, hsv[..., 2])
  hsv[..., 2] = np.where(hsv[..., 2] <= lim, lim+value, hsv[..., 2])

  z = hsv.reshape((-1, 3))
  Z = np.float32(z)
  cr = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  k = 2
  ret, label, center = cv2.kmeans(
      Z, k, None, cr, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  res = np.array(center[label.flatten()])
  res = res.reshape((hsv.shape))

  rgb = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
  binary = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
  h_mask = np.where(binary <= int(binary.mean()), 1, binary)
  h_mask = np.where(h_mask != 1, 0, h_mask)
  new_img = pcv.apply_mask(img=rgb_img, mask=h_mask, mask_color='white')
  gray_image = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)

  return h_mask, gray_image, new_img


def mask_a_pixels(rgb_img):
  a_mask = pcv.rgb2gray_lab(rgb_img=rgb_img, channel='a')
  a_mask = np.where(a_mask <= int(a_mask.mean()), 1, a_mask)
  a_mask = np.where(a_mask != 1, 0, a_mask)
  new_img = pcv.apply_mask(img=rgb_img, mask=a_mask, mask_color='white')
  gray_image = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
  return a_mask, gray_image, new_img


def mask_green_pixels(rgb_img):
  r_channel, g_channel, b_channel = cv2.split(rgb_img)
  # Set those pixels where green value is larger than both blue and red to 0
  # True is treated as 1 and False as 0
  mask = True == np.multiply(g_channel > r_channel, g_channel > b_channel)
  mask = mask.astype(np.uint8)
  new_img = pcv.apply_mask(img=rgb_img, mask=mask, mask_color='white')
  gray_image = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
  return mask, gray_image, new_img

bgr_img = cv2.imread('path_img')
rgb_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
g_blurred = cv2.GaussianBlur(rgb_image, (5, 5), 0)
g_mask, ggray_img, rgb_im = mask_green_pixels(g_blurred)
a_mask, gray_img, new_rgb_img = mask_a_pixels(g_blurred)
h_mask, hgray_img, h_rgb_img = mask_h_pixels(g_blurred)
cnts = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
h_mask = cv2.fillPoly(h_mask, cnts, (255,255,255))
no_back_img = cv2.bitwise_and(rgb_image, rgb_image, mask=h_mask)
image_gray = pcv.threshold.gaussian(gray_img, 255, 'dark')
image_gray = pcv.rgb2gray_hsv(rgb_img=rgb_image, channel='s')