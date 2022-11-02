import os
import PIL
import cv2
import glob
import h5py
import json
import wandb
import random
import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.subplots as sp
from PIL import Image
from transformers import ViTFeatureExtractor, ConvNextFeatureExtractor, AutoFeatureExtractor, DefaultDataCollator

RANDOM_SEED = 42


def store_hdf5(name, train_x, valid_x, test_x, train_y, valid_y, test_y):
    """
    Stores an array of images to HDF5.

    Args:
        name(str):				filename
        train_x(numpy.array):   training images array
        valid_x(numpy.array): 	validation images array
        test_x(numpy.array):  	testing images array
        train_y(numpy.array): 	training labels array
        valid_y(numpy.array): 	validation labels array
        test_y(numpy.array): 	testing labels array
    Returns:
        file(h5py.File): file containing
    """

    # Create a new HDF5 file
    file = h5py.File(name, "w")
    print(f"Train Images:     {np.shape(train_x)}  -- dtype: {train_x.dtype}")
    print(f"Train Labels:    {np.shape(train_y)} -- dtype: {train_y.dtype}")

    # Images are store as uint8 -> 0-255
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


def load_hdf5(name, class_label, label_only=False):
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
    if label_only:
        return labels
    else:
        return images, labels


class PlantDataset():
    """
    Class to represents our plant dataset with useful metadata and
    statistics.

    In the basefolder, the seperate folders containing the images must
    have the format {plant}___{plant_disease}.

    Attributes:
        basefolder (bytes): path to the directory containing our images
        shape (tuple): expected shape of our images
        max_samples_per_class (int): maximum sample per class for balancing
        verbose (bool): option to display debugging messages
    """

    def __init__(self, basefolder, shape=(256, 256, 3), max_samples_per_class=1000, verbose=False):
        self.basefolder = basefolder
        self.img_shape = shape
        self.max_samples_per_class = max_samples_per_class
        self.verbose = verbose


    def load_data(self, seed=42, dtype='float32'):
        """
        Stores an array of images to HDF5.

        Returns:
            plant_df (pandas.DataFrame): recapitulatory dataframe of our dataset
        """
        df_lst = []
        images_lst, healthy_lst, plant_lst, disease_lst = [], [], [], []
        general_disease_lst = []

        total_pic = 0
        classes_folder = os.listdir(f"{self.basefolder}/")
        classes_folder.sort()

        if self.verbose:
            print(classes_folder)
        if seed is not None:
            random.seed(seed)

        if 'Background_without_leaves' in classes_folder:
            classes_folder.remove('Background_without_leaves')

        for i, folder in enumerate(classes_folder):
            class_img_paths = glob.glob(
                os.path.join(self.basefolder, folder, '*'))
            random.shuffle(class_img_paths)

            split_file = folder.split('___')
            plant_specie = split_file[0].lower()
            if ',' in plant_specie:
                plant_specie = plant_specie.replace(',', '')

            if (len(split_file) > 1):
                disease = split_file[1].lower()
                state_img = dict()
                if disease == 'healthy':
                    state_img["healthy"] = 1
                else:
                    state_img["healthy"] = 0
                plant_state = f"{plant_specie}_{disease}"
                # class: healthy -> binary
                healthy = state_img["healthy"]

                img_folder_name = f"{self.basefolder}/{folder}"
                if self.verbose:
                    print('Loading '+img_folder_name)
                img_list = os.listdir(img_folder_name)
                img_list = [i for i in img_list if not i.startswith(".DS")]
                random.shuffle(img_list)
                p_img_lst = []
                for img_file in img_list:
                    absolute_file_name = img_folder_name+'/'+img_file
                    print(f"filename : {absolute_file_name}")
                    image = cv2.imread(absolute_file_name)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #(image, cv2.COLOR_BGR2LAB)
                    if (image.shape != self.img_shape):
                        if (image.shape[-1] != self.img_shape[-1]):
                            print(
                                f"Found wrong image shape: {absolute_file_name}.\nShape {image.shape} instead of {self.img_shape}")
                        if (image.shape[0] > self.img_shape[0]):
                            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

                    images_lst.append(image)
                    healthy_lst.append(healthy)
                    plant_lst.append(plant_specie)
                    disease_lst.append(plant_state)
                    general_disease_lst.append(disease)
                    p_img_lst.append(absolute_file_name)
                state_img['plant'] = plant_specie
                state_img['labels'] = plant_state
                state_img['label_count'] = len(p_img_lst)
                state_img['image_path'] = img_folder_name
                new = pd.DataFrame.from_dict([state_img])
                total_pic += len(p_img_lst)
                df_lst.append(new)

        plant_df = pd.concat(df_lst)
        plant_df.reset_index(drop=True, inplace=True)
        if self.verbose == True:
            print(f"\n\nOur Dataset:\n{plant_df}\n")
            print(f"\nNumber of pictures : {total_pic}\n")
            print(
                f"{len(plant_df['plant'].unique())} unique plants :\n{plant_df['plant'].unique()}\n")
            print(
                f"{len(plant_df['labels'].unique())} unique classes :\n{plant_df['labels'].unique()}")
            print(f'Image Type: {type(image)}')
            print(f"Image dtype: {image.dtype}")
            print(f"Image Size: {image.size}")
            print(f"Image nb bytes: {image.nbytes}")
            print(f"Image strides: {image.strides}")
            print(f"Image Shape: {image.shape}")
            print(f"Max RGB Value: {image.max()}")
            print(f"Min RGB Value: {image.min()}")
            print(
                f"RGB values for pixel (100th rows, 50th column): {image[100, 50]}\n")


        for x in images_lst:
            if x.dtype != 'uint8':
                print(f"image is {x.dtype}")

        images_arr = np.array(images_lst)#, dtype=np.uint8)
        healthy_arr = np.array(healthy_lst).astype(int).astype(bool)

        plants_d = {i: np.unique(plant_lst)[i]
                    for i in range(len(np.unique(plant_lst)))}
        diseases_d = {i: np.unique(disease_lst)[i]
                      for i in range(len(np.unique(disease_lst)))}
        general_diseases_d = {i: np.unique(general_disease_lst)[i]
                              for i in range(len(np.unique(general_disease_lst)))}

        labels_dict = {"plants": plants_d, "diseases": diseases_d,
                       "general_diseases": general_diseases_d}
        #for name, elem in labels_dict.items():
        #    with open(f'../resources/{name}_label_map.json', 'w') as f:
        #        json.dump(elem, f, indent=4)

        rev_plants_d = {v: k for k, v in plants_d.items()}
        rev_disease_d = {v: k for k, v in diseases_d.items()}
        rev_general_disease_d = {v: k for k, v in general_diseases_d.items()}

        plants_arr = np.array([rev_plants_d[val] for val in plant_lst])
        disease_arr = np.array([rev_disease_d.get(val) for val in disease_lst])
        general_disease_arr = np.array(
            [rev_general_disease_d.get(val) for val in general_disease_lst])

        self.img_nbr = total_pic
        self.images = images_arr
        self.healthy = healthy_arr
        self.plants = plants_arr
        self.diseases = disease_arr
        self.general_diseases = general_disease_arr
        return plant_df


    def get_relevant_images_labels(self, label_type):
        """" Returns the corresponding labels given the label_type. """
        if label_type == 'plant':
            label = self.plants
        elif label_type == 'disease':
            label = self.diseases
        elif label_type == 'gen_disease':
            label = self.gen_diseases
        elif label_type == 'healthy':
            label = self.healthy
        return self.images, label


    def dataset_distribution(self, plant_df):
        """
        Displays a histogram and a pie chart for each category
        (plant/healthy/disease) of our dataset.

        Args:
            plant_df (pandas.DataFrame): dataframe of our dataset
        """
        cols = ['healthy', 'plant', 'labels']
        for col in cols:
            if col != 'labels':
                elem_cnt = []
                for elem in plant_df[col].unique():
                    subset_df = plant_df[plant_df[col] == elem]
                    count = (elem, subset_df['label_count'].sum())
                    elem_cnt.append(count)
                elem_cnt = sorted(elem_cnt, key=lambda x: x[1])
                names = [x[0] for x in elem_cnt]
                cnts = [x[1] for x in elem_cnt]
                figure1 = px.pie(names, names=names,
                                 values=cnts, color=names)
                figure2 = px.histogram(
                    names, x=names, y=cnts, color=names)
            else:
                plant_df = plant_df.sort_values('label_count')
                figure1 = px.pie(plant_df, names=col,
                                 values='label_count', color=col)
                figure2 = px.histogram(
                    plant_df, x=col, y='label_count', color=col)

            figure1_traces = []
            figure2_traces = []
            for trace in range(len(figure1["data"])):
                figure1_traces.append(figure1["data"][trace])
            for trace in range(len(figure2["data"])):
                figure2_traces.append(figure2["data"][trace])

            this_figure = sp.make_subplots(
                rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "xy"}]])

            for traces in figure1_traces:
                this_figure.append_trace(traces, row=1, col=1)
            for traces in figure2_traces:
                this_figure.append_trace(traces, row=1, col=2)

            this_figure.update_layout(barmode="stack", height=600, width=1700,
                                      title_text=f"{col} distribution across dataset", showlegend=False)
            this_figure.show()
            this_figure.write_image("../resources/metadata/ds_labels_distrib.jpeg")


    def plant_overview(self, plant_df):
        """
        Displays a histogram and a pie chart for each plant.

        Args:
            plant_df (pandas.DataFrame): dataframe of our dataset
        """
        unique_plant = plant_df['plant'].unique()
        for elem in unique_plant:
            subset_df = plant_df[plant_df['plant'] == elem]
            subset_df = subset_df.sort_values('label_count', ascending=False)

            # Create figures in Express
            figure1 = px.pie(
                subset_df, names='labels', values='label_count', color='labels')
            # .update_yaxes(categoryorder='total descending')
            figure2 = px.histogram(
                subset_df, x='labels', y='label_count', color='labels')

            # For as many traces that exist per Express figure, get the traces
            # from each plot and store them in an array.
            figure1_traces = []
            figure2_traces = []
            for trace in range(len(figure1["data"])):
                figure1_traces.append(figure1["data"][trace])
            for trace in range(len(figure2["data"])):
                figure2_traces.append(figure2["data"][trace])

            # Create a 1x2 subplot
            this_figure = sp.make_subplots(
                rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "xy"}]])

            # Get the Express fig broken down as traces and add the traces to the proper plot within in the subplot
            for traces in figure1_traces:
                this_figure.append_trace(traces, row=1, col=1)
            for traces in figure2_traces:
                this_figure.append_trace(traces, row=1, col=2)

            # the subplot as shown in the above image
            this_figure.update_layout(barmode="stack", height=600, width=1700,
                                      title_text=f"{elem} class distribution", showlegend=False)
            this_figure.show()
            this_figure.write_image(f"../resources/metadata/ds_{elem}_distrib.jpeg")


def resize_images(img_arr, img_size):
    resized_arr = []
    for img in img_arr:
        new_img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_AREA)
        resized_arr.append(new_img)
    resized_arr = np.array(resized_arr)
    return resized_arr

def process(examples, feature_extractor):
    """" Maps the feature_extractor to our image array """
    examples.update(feature_extractor(examples['img'], ))
    return examples

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['img']], return_tensors='tf')
    # Don't forget to include the labels!
    inputs['labels'] = example_batch['labels']
    return inputs

def create_hf_ds(images, labels, feature_extractor, class_names):
    """" Creates a dataset with transformer feature exctractor """
    features = datasets.Features({
        "img": datasets.Image(),
        # ClassLabel feature type is for single-label multi-class classification
        # For multi-label classification (after one hot encoding) you can use Sequence with ClassLabel
        "label": datasets.features.ClassLabel(names=class_names)
    })

    print(features)
    ds = datasets.Dataset.from_dict(
        {"img": images, "label": labels}, features=features)
    ds = ds.rename_column("label", "labels")

    #ds = ds.map(lambda x: process(x, feature_extractor), batched=True)#, writer_batch_size=10)
    ds = ds.with_transform(transform)

    ds = ds.shuffle(seed=42)
    return ds

def create_transformer_ds(label_type, X_train, X_valid, X_test, y_train, y_valid, y_test):
    """
    Creates a training, validation and test set dataset (given the label type) with the appropriate feature
    extractor applied for each of the ViT, Swin and ConvNexT models.
    """
    if label_type !=  'healthy':
        if label_type == 'disease':
            label_map_path = '../resources/label_maps/diseases_label_map.json'
        elif label_type == 'plants':
            label_map_path = '../resources/label_maps/plants_label_map.json'
        else:
            label_map_path = '../resources/label_maps/general_diseases_label_map.json'
        with open(label_map_path) as f:
            id2label = json.load(f)
        class_names = [str(v) for k,v in id2label.items()]
    else:
        class_names = ['healthy', 'not_healthy']

    fe_dict = {
        'vit': ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224"),
        'swin': AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224"),
        'convnext': ConvNextFeatureExtractor.from_pretrained("facebook/convnext-tiny-224")
    }
    for name, feature_extractor in fe_dict.items():
        train_sets = create_hf_ds(
            X_train, y_train, feature_extractor, class_names)
        train_sets.save_to_disk(
            f"../resources/datasets/transformers/{name}/train")

        valid_sets = create_hf_ds(
            X_valid, y_valid, feature_extractor, class_names)
        valid_sets.save_to_disk(
            f"../resources/datasets/transformers/{name}/valid")

        test_sets = create_hf_ds(
            X_test, y_test, feature_extractor, class_names)
        test_sets.save_to_disk(
            f"../resources/datasets/transformers/{name}/test")
