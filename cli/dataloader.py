import os
import h5py
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import plotly.express as px
import plotly.subplots as sp
import random
import glob
import wandb
import json
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import torch
from torch import nn

RANDOM_SEED = 42

def store_hdf5(name, images, healthy, plant, disease, gen_disease):
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
    print(f"Images:     {np.shape(images)}  -- dtype: {images.dtype}")
    print(f"Healthy:    {np.shape(healthy)} -- dtype: {healthy.dtype}")
    print(f"Plant:      {np.shape(plant)}   -- dtype: {plant.dtype}")
    print(f"Disease:    {np.shape(disease)} -- dtype: {disease.dtype}")
    print(
        f"Gen_Disease: {np.shape(gen_disease)} -- dtype: {gen_disease.dtype}")

    # Create an image dataset in the file
    # store as uint8 -> 0-255
    img_dataset = file.create_dataset("images", np.shape(
        images), h5py.h5t.STD_U8BE, data=images)

    # Create a label dataset for healthy/sick in the file
    healthy_set = file.create_dataset("healthy", np.shape(
        healthy), h5py.h5t.STD_U8BE, data=healthy)
    # Create a label dataset for plants in the file
    plant_set = file.create_dataset("plant", np.shape(
        plant), h5py.h5t.STD_U8BE, data=plant)
    # Create a label dataset for diseases in the file
    disease_set = file.create_dataset("disease", np.shape(
        disease), h5py.h5t.STD_U8BE, data=disease)
    # Create a label dataset for general diseases in the file
    gen_disease_set = file.create_dataset("gen_disease", np.shape(
        gen_disease), h5py.h5t.STD_U8BE, data=gen_disease)

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
    Class to represents the PlantVillage dataset with useful metadata and
    statistics.

    Attributes:
        basefolder (bytes): path to the directory containing our images
        shape (tuple): expected shape of our images
        max_samples_per_class (int): maximum sample per class for balancing
        verbose (bool): option to display debugging messages
    """

    #def __init__(self, basefolder, shape=(224, 224, 3), max_samples_per_class=1000, verbose=False):
    def __init__(self, basefolder, shape=(256, 256, 3), max_samples_per_class=1000, verbose=False):
        self.basefolder = basefolder
        self.img_shape = shape
        self.max_samples_per_class = max_samples_per_class
        self.verbose = verbose


    def load_data(self, seed=42, dtype='float32', balanced=False,):
        """
        Stores an array of images to HDF5.

        Args:
            balanced (boolean): option to balance the classes of our dataset
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

                healthy = state_img["healthy"]
                img_folder_name = f"{self.basefolder}/{folder}"
                if self.verbose:
                    print('Loading '+img_folder_name)
                img_list = os.listdir(img_folder_name)
                img_list = [i for i in img_list if not i.startswith(".DS")]
                random.shuffle(img_list)

                #train_path, val_path, test_path, train_y, val_y, test_y = split_for_training(i, class_img_paths, train_path, val_path, test_path, train_y, val_y, test_y)
                #print(f"{folder}\ntrain: {len(train_path)}\nval: {len(val_path)}\ntest: {len(test_path)}")
                #train_x = np.array(cai.datasets.load_images_from_files(train_path, target_size=target_size, smart_resize=smart_resize, lab=lab, rescale=True, bipolar=bipolar), dtype='float32')

                p_img_lst = []
                for img_file in img_list:
                    absolute_file_name = img_folder_name+'/'+img_file
                    #print(absolute_file_name)
                    image = cv2.imread(absolute_file_name)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    if (image.shape != self.img_shape):
                        if (image.shape[-1] != self.img_shape[-1]):
                            print(
                                f"Found wrong image shape: {absolute_file_name}.\nShape {image.shape} instead of {self.img_shape}")
                        if (image.shape[-1] > self.img_shape[-1]):
                            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
                    ##continue
                    #raw_data_at.add_file()

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
                if balanced == True:
                    if len(p_img_lst) > self.max_samples_per_class:
                        print("Reduce this class")
                        new = new.sample(
                            self.max_samples_per_class, random_state=RANDOM_SEED)
                        new.reset_index(drop=True, inplace=True)
                        state_img['label_count'] = self.max_samples_per_class
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

        images_arr = np.array(images_lst, dtype=np.uint8)
        healthy_arr = np.array(healthy_lst).astype(int).astype(bool)

        # {"label":id} -> reversed
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
        rev_diseases_d = {v: k for k, v in diseases_d.items()}
        rev_general_diseases_d = {v: k for k, v in general_diseases_d.items()}

        plants_arr = np.array([rev_plants_d[val] for val in plant_lst])
        diseases_arr = np.array([rev_diseases_d.get(val)
                                for val in disease_lst])
        general_diseases_arr = np.array(
            [rev_general_diseases_d.get(val) for val in general_disease_lst])

        self.img_nbr = total_pic
        self.dataset_path = f'dataset_224_{self.img_nbr}.h5'
        self.images = images_arr
        self.healthy = healthy_arr
        self.plants = plants_arr
        self.diseases = diseases_arr
        self.general_diseases = general_diseases_arr
        #self.dataset = store_hdf5(self.dataset_path, images_arr, healthy_arr,
        #                          plants_arr, diseases_arr, general_diseases_arr)
        return plant_df

    def get_relevant_images_labels(self, label_type):
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
