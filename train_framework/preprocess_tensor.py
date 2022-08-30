import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
#from keras.preprocessing.image import ImageDataGenerator
#from framework.utils import *
import multiprocessing
import gc

@tf.function
def get_mean_std(train_set):
    """
    Calculate metric on the training set for normalization/image processing.
    These metrics are the mean and the standard deviation for each
    color channel (RGB).
    """
    train_set = tf.cast(train_set, tf.float64)
    tf_std = tf.math.reduce_std(train_set, [0, 1, 2])
    tf_mean = tf.math.reduce_mean(train_set, [0, 1, 2])
    tf.print(f"TRAIN MEAN (TF): {tf_mean}")
    tf.print(f"TRAIN STD (TF): {tf_std}")
    return tf_mean, tf_std

@tf.function
def resize_img(img, label):
    img = tf.image.resize(img, (299, 299))
    return img, label


@tf.function
def encode_categorical(img, label, n_classes):
    label = tf.one_hot(label, n_classes,  dtype='uint8')
    return img, label

@tf.function
def to_vector(img, label):
    label = tf.expand_dims(label, axis=1)
    return img, label

#@tf.function
def prep_ds_input(args, ds, set_len):
    N_CPUS = multiprocessing.cpu_count()
    print(f"NBR CPUS: {N_CPUS}")
    if args.transformer:
        #ds = ds.map(lambda elem, label: to_vector(
        #            elem, label), num_parallel_calls=N_CPUS)
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda elem, label: encode_categorical(
                    elem, label, args.n_classes), num_parallel_calls=N_CPUS)
        ds = ds.shuffle(set_len, seed=args.seed).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def preprocess_image(tensor_img, mode='centering'):
    """Preprocesses a Numpy array encoding a batch of images.
    Args:
      tensor_img: Input tensor, 3D or 4D.

      mode:
        - centering: will convert the images from RGB to BGR,
            then will zero-center each color channel with
            respect to the training dataset, without scaling.
        - sample_wise_scaling: will scale pixels between -1 and 1, sample-wise.
        - scale_std: will scale pixels between 0 and 1 then zero-center
            by mean and finally normalize each channel with respect to the
            training dataset.
    Returns:
        Preprocessed tensor.
    """
    if mode == None:
        return tensor_img

    if type(mode) != str:
        tensor_img = mode(tensor_img)
        return tensor_img

    tensor_img = tf.cast(tensor_img, tf.float32, name=None)

    #mean_arr = [118.94, 124.72, 104.59]
    #std_arr = [49.35, 42.97, 54.13]
    #augm_mean_arr = [118.14, 124.61, 104.01]
    #augm_std_arr = [49.30, 42.62, 54.95]
    #lab_augm_mean_arr = [129.75, 122.14, 138.48]
    #lab_augm_std_arr = [44.66, 12.08, 15.12]

    augm_mean_arr_128 = [118.26, 124.73, 104.13]
    augm_std_arr_128 = [48.91, 42.17, 53.59]

    train_mean = tf.convert_to_tensor(augm_mean_arr_128, dtype=tf.float32)
    train_std = tf.convert_to_tensor(augm_std_arr_128, dtype=tf.float32)
    # TRANSFORMERS MODELS -> apply same preprocessing -> size = 224
    ## IMAGENET WEIGHTS -> when using pretrained models 
    #train_mean = tf.convert_to_tensor([123.675, 116.28, 103.53], dtype=tf.float32)
    #train_std = tf.convert_to_tensor([58.395, 57.12, 0.225], dtype=tf.float32)
    

    data_format = K.image_data_format()
    assert data_format == 'channels_last'

    if mode == 'sample_wise_scaling':
        print("scale pixels between -1 and 1")
        tensor_img /= 127.5
        tensor_img -= 1.
        return tensor_img

    elif mode == 'scale_to_floats':
        # for faster computation only (does not change spread of pixel values)
        # RGB has already an abounding system of 0-255
        print("scale pixels between 0 and 1")
        tensor_img /= 255.
        return tensor_img

    elif mode == 'scale_std':
        print("Scale between 0 and 1 and stardardize pixels from train set stats")
        tensor_img /= 255.
        mean = train_mean/255
        std_tensor = train_std/255

    else:
        print("Centering pixels from train_set mean")
        mean = train_mean
        std_tensor = None

    # convert all values in mean to negative values
    neg_mean_tensor = -mean
    # Zero-center by mean pixel
    if K.dtype(tensor_img) != K.dtype(neg_mean_tensor):
      tensor_img = K.bias_add(
          tensor_img, K.cast(neg_mean_tensor, K.dtype(tensor_img)), data_format=data_format)
    else:
      tensor_img = K.bias_add(tensor_img, neg_mean_tensor, data_format)

    # normalize by std
    if std_tensor is not None:
      tensor_img /= std_tensor

    return tensor_img


def check_preprocessing(args, validation_set, cast_type):
    for elem, label in validation_set.take(1):
        img = elem[0].numpy()
        print(f"image shape is {img.shape}")
        print(f"img 1st pixel of Red channel: {img[0, 0, 0]}")
        # calculate mean value from RGB channels and flatten to 1D array
        vals = img.mean(axis=2).flatten()
        # plot histogram with 255 bins
        b, bins, patches = plt.hist(vals)
        plt.show()

        # TF image resize returns floats -> cast to uint to
        plt.imshow(tf.cast(img, cast_type))
        plt.show()
