import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from train_framework.utils import bcolors, logging

logger = logging.getLogger(__name__)


def load_hdf5(name, class_label):
    """
    Reads image from HDF5.

    Args:
        name(str):              path to the HDF5 file (dataset)
        class_label(str):       type of classification
    Returns:
        images(numpy.array):    images array, (N, 256, 256, 3) to be stored
        labels(numpy.array):    labels array, (N,) to be stored
    """
    images, labels = [], []
    # Open the HDF5 file
    file = h5py.File(f"{name}", "r+")
    # images are stored as uint8 -> 0-255
    images = np.array(file["/images"]).astype(np.uint8)

    if class_label == 'healthy':
        labels = np.array(file["/healthy"]).astype(np.uint8)
    elif class_label == 'plant':
        labels = np.array(file["/plant"]).astype(np.uint8)
    elif class_label == 'disease':
        labels = np.array(file["/disease"]).astype(np.uint8)
    elif class_label == 'gen_disease':
        labels = np.array(file["/gen_disease"]).astype(np.uint8)
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


def get_split_sets(args, images, labels, logger):
    """
    Prepare the inputs and target split sets for a given classification.

    Args:
        args(ArgumentParser): Object that holds multiple training parameters
        images(numpy.array): images array
        labels(numpy.array): label array
        logger():
    Returns:
        X_splits(list): list containing the images split sets
        y_splits(list): list containing the labels split sets
    """

    # Split train, valid, test
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        images, labels, test_size=0.20, stratify=labels, random_state=args.seed)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=args.seed)

    del X_tmp
    del y_tmp
    label_split_lst = list([y_train, y_valid, y_test])
    name_lst = list(['Train', 'Valid', 'Test'])

    # Display counts for unique values in each set
    for value, d_set in zip(label_split_lst, name_lst):
        (unique, cnt) = np.unique(value, return_counts=True)
        logger.info(f"  {d_set} Labels:")
        for name, counts in zip(unique, cnt):
            logger.info(f"    {name} = {counts}")
        if args.class_type == 'healthy':
            logger.info(f"  Ratio Healthy = {cnt[0]/(cnt[0]+cnt[1])}")
            logger.info(f"  Ratio Sick = {cnt[1]/(cnt[0]+cnt[1])}\n")

    if args.class_type == 'healthy':
        y_train = y_train[:, np.newaxis]
        y_valid = y_valid[:, np.newaxis]
        y_test = y_test[:, np.newaxis]

    return [X_train, X_valid, X_test], label_split_lst


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


def get_relevant_datasets(args, logger):
    """
    Get the relevant datasets for a given label.

    Args:
        args(ArgumentParser): Object that holds multiple training parameters
        logger():
    Returns:
        X_splits(list): list containing the images split sets
        y_splits(list): list containing the labels split sets
        n_classes(int): number of unique class in the dataset
    """
    images, labels = load_hdf5(args.dataset, args.class_type)

    if args.class_type not in ['plant', 'disease', 'healthy', 'gen_disease']:
        raise ValueError(f'Classification {args.class_type} not found')
    else:
        (unique_labels, cnt) = np.unique(labels, return_counts=True)
        n_classes = len(unique_labels)
        for name, counts in zip(unique_labels, cnt):
            print(
                f"{bcolors.OKBLUE}{name}{bcolors.ENDC} = {bcolors.OKGREEN}{counts}{bcolors.ENDC}")
        if args.class_type == 'healthy':
            logger.info(f"  Ratio Healthy = {cnt[0]/(cnt[0]+cnt[1])}")
            logger.info(f"  Ratio Sick = {cnt[1]/(cnt[0]+cnt[1])}\n")

        X_splits, y_splits = get_split_sets(args, images, labels, logger)
        return X_splits, y_splits, n_classes
