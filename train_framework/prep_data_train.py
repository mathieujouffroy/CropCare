import h5py
import numpy as np
import json
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import datasets
from transformers import ViTFeatureExtractor
from transformers import DefaultDataCollator
from utils import *

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



def store_hdf5(name, train_x, valid_x, test_x, train_y, valid_y, test_y):
    """
    Stores an array of images to HDF5.

    Args:
    images(numpy.array):    images array, (N, 256, 256, 3) to be stored
    healthy(numpy.array):   healthy/sick (labels) array, (N, 1) to be stored
    plant(numpy.array):     plant (labels) array, (N, 1) to be stored
    disease(numpy.array):   disease (labels) array, (N, 1) to be stored
    gen_disease(numpy.array): general disease (labels) array, (N, 1) to be stored
    """

    # Create a new HDF5 file
    file = h5py.File(name, "w")
    print(f"Train Images:     {np.shape(train_x)}  -- dtype: {train_x.dtype}")
    print(f"Train Labels:    {np.shape(train_y)} -- dtype: {train_y.dtype}")

    # Create an image dataset in the file
    # store as uint8 -> 0-255
    file.create_dataset("train_images", np.shape(train_x),
                        h5py.h5t.STD_U8BE, data=train_x)
    file.create_dataset("valid_images", np.shape(valid_x),
                        h5py.h5t.STD_U8BE, data=valid_x)
    file.create_dataset("test_images", np.shape(test_x),
                        h5py.h5t.STD_U8BE, data=test_x)

    file.create_dataset("train_labels", np.shape(train_y),
                        h5py.h5t.STD_U8BE, data=train_y)
    file.create_dataset("valid_labels", np.shape(valid_y),
                        h5py.h5t.STD_U8BE, data=valid_y)
    file.create_dataset("test_labels", np.shape(test_y),
                        h5py.h5t.STD_U8BE, data=test_y)
    file.close()
    return file


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

feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k")

# basic processing (only resizing)
def process(examples):
    examples.update(feature_extractor(examples['img'], ))
    return examples

def create_hf_ds(images, labels, class_names):
    
    features = datasets.Features({
        "img": datasets.Image(),
        # ClassLabel feature type is for single-label multi-class classification
        # For multi-label classification (after one hot encoding) you can use Sequence with ClassLabel
        "label": datasets.features.ClassLabel(names=class_names)
    })

    print(features['label'])
    print("building dataset")
    
    ds = datasets.Dataset.from_dict(
        {"img": images, "label": labels}, features=features)

    # TEST : 'facebook/deit-base-patch16-224'
    # swin -> microsoft/swin-tiny-patch4-window7-224
    data_collator = DefaultDataCollator(return_tensors="tf")

    ds = ds.rename_column("label", "labels")
    print("before mapping")
    ds = ds.map(process, batched=True)#, writer_batch_size=10)
    print("before shuffle")
    ds = ds.shuffle(seed=42)
    print("after mapping")
    tf_dataset = ds.to_tf_dataset(
       columns=['pixel_values'],
       label_cols=["labels"],
       shuffle=True,
       batch_size=32,
       collate_fn=data_collator)
    return tf_dataset
    #return ds