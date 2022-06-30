import h5py
import numpy as np
import json
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils import *

logger = logging.getLogger(__name__)

def load_split_hdf5(name, split_set):
    """
    Reads image from HDF5.

    Args:
        name(str):              path to the HDF5 file (dataset)
        split_set(str):       type of classification
    Returns:
        images(numpy.array):    images array, (N, 256, 256, 3) to be stored
        labels(numpy.array):    labels array, (N,) to be stored
    """
    images, labels = [], []
    # Open the HDF5 file
    file = h5py.File(f"{name}", "r+")
    # images are stored as uint8 -> 0-255

    if split_set == 'train':
        images = np.array(file["/train_images"]).astype(np.uint8)
        labels = np.array(file["/train_labels"]).astype(np.uint8)
    elif split_set == 'valid':
        images = np.array(file["/valid_images"]).astype(np.uint8)
        labels = np.array(file["/valid_labels"]).astype(np.uint8)
    elif split_set == 'test':
        images = np.array(file["/test_images"]).astype(np.uint8)
        labels = np.array(file["/test_labels"]).astype(np.uint8)
    return images, labels

def prepare_categorical_targets(y_train, y_valid, y_test):
    """
    Prepare categorical targets for training.

    Args:
        y_train(numpy.array):  train labels
        y_valid(numpy.array):  valid labels
        y_test(numpy.array):   test labels
    Returns:
        y_train_enc(numpy.array):  encoded train labels
        y_valid_enc(numpy.array):  encoded valid labels
        y_test_enc(numpy.array):   encoded test labels
        le(sklearn.preprocessing.LabelEncoder):  label encoder
    """
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_valid_enc = le.transform(y_valid)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_valid_enc, y_test_enc, le


def generate_class_weights(y_train, class_type, logger):
    """
    Generate the class weights for a given classification.
    Class weights are used for imbalance classification.
    wj = n_samples / (n_classes * n_samplesj)
    WJ -> weight of each class (J is the class)

    Args:
        y_train(numpy.array): array of labels
        class_type(str): type of classification
    Returns:
        dictionary containing the labels (classes) and their weights
    """
    if class_type != 'healthy':
        class_labels = np.unique(y_train)
        logger.info(f"  labels: {class_labels}")
        unique_class_weights = compute_class_weight(
            class_weight='balanced', classes=class_labels, y=y_train)
    else:
        # Binary classification
        class_labels = np.unique(y_train)
        (unique, counts) = np.unique(y_train, return_counts=True)
        logger.info(f"\n  Y_data values : {unique}")
        logger.info(f"  Y_data counts : {counts}")
        train_neg = counts[0]
        train_pos = counts[1]
        train_tot = train_neg + train_pos
        weight_for_0 = (1 / train_neg) * (train_tot / 2.0)
        weight_for_1 = (1 / train_pos) * (train_tot / 2.0)
        class_w = {0: weight_for_0, 1: weight_for_1}
        logger.info(f"  Manual calculated class weights : {class_w}")
        unique_class_weights = compute_class_weight(
            class_weight='balanced', classes=class_labels, y=y_train)
    logger.info(f"  SK class weights : {unique_class_weights}")
    return dict(zip(class_labels, unique_class_weights))
