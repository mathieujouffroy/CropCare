import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
#from keras.preprocessing.image import ImageDataGenerator
#from framework.utils import *
import multiprocessing
import datasets
from transformers import ViTFeatureExtractor
from transformers import DefaultDataCollator


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
    img = tf.image.resize(img, (128, 128))
    return img, label


@tf.function
def encode_categorical(img, label, n_classes):
    label = tf.one_hot(label, n_classes,  dtype='uint8')
    return img, label


@tf.function
def prepare_input_shape(img, label, n_classes):
    """
    Prepare input shape for the model.
    """
    #if img.shape[0] != 224:
    #    img = tf.image.resize(img, (224, 224))
    label = tf.one_hot(label, n_classes,  dtype='uint8')
    return img, label


#@tf.function
def prep_ds_input(args, ds, set_type):
    print(f"NBR TRAIN EXAMPLES: {args.len_train}")
    N_CPUS = multiprocessing.cpu_count()
    print(f"NBR CPUS: {N_CPUS}")
    if set_type == 'train':
        ds = ds.map(lambda elem, label: prepare_input_shape(
            elem, label, args.n_classes), num_parallel_calls=N_CPUS)\
            .shuffle(args.len_train, seed=args.seed)
    else:
        ds = ds.map(lambda elem, label: prepare_input_shape(
            elem, label, args.n_classes), num_parallel_calls=N_CPUS)\
            .shuffle(args.len_valid, seed=args.seed)

    if args.transformer:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

        for elem, label in ds.take(1):
            print(f"elem shape is: {elem.shape}")
            print(f"label shape is: {label.shape}")

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

    train_mean = tf.convert_to_tensor(
        [118.933, 124.707, 104.610], dtype=tf.float32)
    train_std = tf.convert_to_tensor(
        [50.728, 44.546, 55.380], dtype=tf.float32)
    tensor_img = tf.cast(tensor_img, tf.float32, name=None)

    data_format = K.image_data_format()
    assert data_format == 'channels_last'

    if mode == 'no':
        return tensor_img

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


def process(examples):
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    examples.update(feature_extractor(examples['img'], ))
    return examples


def create_hf_ds(args, images, labels):
    features=datasets.Features({
        "img": datasets.Image(),
        # ClassLabel feature type is for single-label multi-class classification
        # For multi-label classification (after one hot encoding) you can use Sequence with ClassLabel
        "label": datasets.features.ClassLabel(names=args.class_names)
    })
    ds = datasets.Dataset.from_dict(
        {"img": images, "label": labels}, features=features)

    # TEST : 'facebook/deit-base-patch16-224'
    # swin -> microsoft/swin-tiny-patch4-window7-224
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    data_collator = DefaultDataCollator(return_tensors="tf")

    ds = ds.rename_column("label", "labels")
    ds = ds.map(process, batched=True)
    #ds = ds.shuffle(seed=args.seed)

    tf_dataset = ds.to_tf_dataset(
       columns=['pixel_values'],
       label_cols=["labels"],
       collate_fn=data_collator,
       batch_size=args.batch_size)

    for elem, label in tf_dataset.take(1):
        img = elem[0].numpy()
        print(f"element shape is {elem.shape}, type is {elem.dtype}")
        print(f"image shape is {img.shape}, type: {img.dtype}")
        print(f"label shape is {label.shape} type: {label.dtype}")

    return tf_dataset
