import tensorflow as tf
import multiprocessing
import matplotlib.pyplot as plt
from keras import backend as K


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
def resize_img(img, label, size):
    """ Resize an image to the give size """

    img = tf.image.resize(img, size)
    return img, label


@tf.function
def prep_inputs_and_labels(img, label, n_classes, size):
    """ Preprocess our inputs and labels. Resizing and one-hot-encoding. """

    img, label = resize_img(img, label, size)
    label = tf.one_hot(label, n_classes,  dtype='uint8')
    return img, label


@tf.function
def to_vector(img, label):
    label = tf.expand_dims(label, axis=1)
    return img, label


def prep_ds_input(args, ds, set_len, size):
    """ Preprocssing function that maps the relevant preprocessing steps. """
    N_CPUS = multiprocessing.cpu_count()
    if args.transformer:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda elem, label: prep_inputs_and_labels(
                    elem, label, args.n_classes, size), num_parallel_calls=N_CPUS)
        ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def preprocess_image(tensor_img, mean_arr, std_arr, mode='centering'):
    """Preprocesses a Numpy array encoding a batch of images.
    Args:
        tensor_img: Input tensor, 3D or 4D.
        mean_arr: Array containing the RGB mean for our dataset.
        std_arr: Array containing the RGB stdev for our dataset.
        mode:
            - centering: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the training dataset, without scaling.
            - sample_wise_scaling (tf): will scale pixels between -1 and 1, sample-wise.
            - scale_std (torch): will scale pixels between 0 and 1 then zero-center
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
    train_mean = tf.convert_to_tensor(mean_arr, dtype=tf.float32)
    train_std = tf.convert_to_tensor(std_arr, dtype=tf.float32)

    data_format = K.image_data_format()
    assert data_format == 'channels_last'

    if mode == 'sample_wise_scaling':
        tensor_img /= 127.5
        tensor_img -= 1.
        return tensor_img

    elif mode == 'scale_to_floats':
        # for faster computation only (does not change spread of pixel values)
        # RGB has already an abounding system of 0-255
        tensor_img /= 255.
        return tensor_img

    elif mode == 'scale_std':
        tensor_img /= 255.
        mean = train_mean/255
        std_tensor = train_std/255

    else:
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
