import sys
import json
from tkinter import N
import wandb
import random
import numpy as np
from cli_utils import bcolors, strawb, animate
from dataloader import PlantDataset, store_hdf5, create_transformer_ds, resize_images
from leaf_segmentation import segment_split_set
from sklearn.model_selection import train_test_split

def get_imgs_table(valid_x, valid_y, nbr_imgs):
    img_dict = dict()
    for img, label in zip(valid_x, valid_y):
        if label not in img_dict.keys():
            img_dict[label] = []
        img_dict[label].append(img)

    imgs_table = []
    i = 0
    with open("resources/label_maps/diseases_label_map.json") as f:
        CLASS_INDEX = json.load(f)
    for unique_class, img_lst in img_dict.items():
        sample = random.sample(img_lst, nbr_imgs)
        for img in sample:
            class_name = CLASS_INDEX[str(unique_class)]
            imgs_table.append([i, class_name, wandb.Image(img)])
            i += 1

    return imgs_table

def viz_dataset_wandb(valid_x, valid_y, nbr_imgs):
    imgs_table = get_imgs_table(valid_x, valid_y, nbr_imgs)
    # Initialize a new W&B run
    run = wandb.init(project='cropdis_vis', group='viz_data', reinit=True)
    # Intialize a W&B Artifacts
    ds = wandb.Artifact("cropdis_ds", type="raw_data")

    # create a wandb.Table() with corresponding columns
    columns = ["id", "label", "image"]

    valid_table = wandb.Table(data=imgs_table, columns=columns)
    run.log({"table_key": valid_table})

    # Add the table to the Artifact
    ds.add(valid_table, 'valid_data')
    run.log_artifact(ds)
    #ds.save()
    wandb.run.finish()


def get_split_sets(seed, class_type, images, labels):
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
        images, labels, test_size=0.30, stratify=labels, random_state=seed)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    label_split_lst = list([y_train, y_valid, y_test])
    name_lst = list(['Train', 'Valid', 'Test'])

    # Display counts for unique values in each set
    for value, d_set in zip(label_split_lst, name_lst):
        (unique, cnt) = np.unique(value, return_counts=True)
        print(f"  {d_set} Labels:")
        for name, counts in zip(unique, cnt):
            print(f"    {name} = {counts}")
        if class_type == 'healthy':
            print(f"  Ratio Healthy = {cnt[0]/(cnt[0]+cnt[1])}")
            print(f"  Ratio Sick = {cnt[1]/(cnt[0]+cnt[1])}\n")

    if class_type == 'healthy':
        y_train = y_train[:, np.newaxis]
        y_valid = y_valid[:, np.newaxis]
        y_test = y_test[:, np.newaxis]

    return [X_train, X_valid, X_test], label_split_lst


def dump_training_stats(X_train, label_type, prefix):
    X_train_mean_rgb = np.round(np.mean(X_train, axis=tuple(range(X_train.ndim-1))), 3)
    X_train_std_rgb = np.round(np.std(X_train, axis=tuple(range(X_train.ndim-1))), 3)
    train_stats = {
        'X_train_mean_rgb': X_train_mean_rgb.tolist(),
        'X_train_std_rgb': X_train_std_rgb.tolist()
    }
    print(f"X train mean : {X_train_mean_rgb}")
    print(f"X train std : {X_train_std_rgb}")
    with open(f"{prefix}{label_type}_train_stats_128.json", "w") as outfile:
        json.dump(train_stats, outfile, indent=4)


def main():
    if len(sys.argv) == 2:
        quit_lst = ['q', 'quit']
        animate()
        plant_data = PlantDataset(sys.argv[1], verbose=True)
        plant_df = plant_data.load_data()
        print(
            f"\n\n{bcolors.OKGREEN}==================  POP FARM : Plant Phenotyping  =================={bcolors.ENDC}")
        print(f"{bcolors.FAIL}{strawb}{bcolors.ENDC}\n\n")

        print(
            f"{bcolors.UNDERLINE}type q or quit to exit the program{bcolors.ENDC}\n")
        label_type = input(f'Enter the label type: plant, disease, healthy, gen_disease\n').lower()
        assert label_type in ['plant', 'disease', 'healthy', 'gen_disease']
        images, labels = plant_data.get_relevant_images_labels(label_type)
        noseg_images = resize_images(images, (128, 128))
        X_splits, y_splits = get_split_sets(42, label_type, noseg_images, labels)
        X_train, X_valid, X_test = X_splits
        y_train, y_valid, y_test = y_splits
        viz_dataset_wandb(X_valid, y_valid, 5)
        # Get stats from training set for data preprocessing
        #dump_training_stats(X_train, label_type, prefix='augm_')
        store_hdf5(f"../resources/datasets/augm_{label_type}_{plant_data.img_nbr}_ds_128.h5", X_train, X_valid, X_test, y_train, y_valid, y_test)
        # CREATE TRANSFORMER DATASET
        create_transformer_ds(label_type, X_train, X_valid, X_test, y_train, y_valid, y_test)
        try:
            while not False:

                options = input(f"""{bcolors.OKBLUE}[0]{bcolors.ENDC} -- Visualization of your farm\n{bcolors.OKBLUE}[1]{bcolors.ENDC} -- Generate Segmented Leaves HSV Mask\n{bcolors.OKBLUE}[2]{bcolors.ENDC} -- Generate Segmented Leaves HSV Mask + dist transform\n{bcolors.OKBLUE}[q]{bcolors.ENDC} -- Quit\n""")

                if options.lower() in quit_lst:
                    print("Bye !")
                    break

                if options == '0':
                    print('Dataset Distribution')
                    plant_data.dataset_distribution(plant_df)
                    plant_data.plant_overview(plant_df)
                    print('Done')

                if options == '1':
                    print('Leag Segmentation HSV mask')
                    p_option = input(
                        f"""Chose Image adjustments (brightness, contrast):\n{bcolors.OKBLUE}[0]{bcolors.ENDC} -- No Adjustments\n{bcolors.OKBLUE}[1]{bcolors.ENDC} -- Adjust Contrast\n{bcolors.OKBLUE}[2]{bcolors.ENDC} -- Adjust Lightness and Contrast\n""")
                    if p_option in ['0', '1', '2']:
                        train_seg = segment_split_set(X_train, p_option)
                        val_seg = segment_split_set(X_valid, p_option)
                        test_seg = segment_split_set(X_test, p_option)
                    else:
                        print('Invalid option')
                        continue

                if options == '2':
                    print('Leag Segmentation HSV mask + dist transform')
                    p_option = input(
                        f"""Chose Image adjustments (brightness, contrast):\n{bcolors.OKBLUE}[0]{bcolors.ENDC} -- No Adjustments\n{bcolors.OKBLUE}[1]{bcolors.ENDC} -- Adjust Contrast\n{bcolors.OKBLUE}[2]{bcolors.ENDC} -- Adjust Lightness and Contrast\n""")
                    if p_option in ['0', '1', '2', '3']:
                        train_seg = segment_split_set(X_train, p_option, dist=True)
                        val_seg = segment_split_set(X_valid, p_option, dist=True)
                        test_seg = segment_split_set(X_test, p_option, dist=True)
                    else:
                        print('Invalid option')
                        continue

                if options in ['1', '2']:
                    dataset_name = f"../resources/datasets/segm_{label_type}_{plant_data.img_nbr}_ds_128.h5"
                    X_train, X_valid, X_test = X_splits
                    y_train, y_valid, y_test = y_splits
                    X_train_seg = resize_images(train_seg, (128, 128))
                    X_val_seg = resize_images(val_seg, (128, 128))
                    X_test_seg = resize_images(test_seg, (128, 128))
                    # Get stats from training set for data preprocessing
                    #dump_training_stats(X_train_seg, label_type, prefix='segm_')
                    store_hdf5(dataset_name,  X_train_seg, X_val_seg, X_test_seg, y_train, y_valid, y_test)

        except EOFError:  # for ctrl + c
          print("\nBye !")
          quit = True
        except KeyboardInterrupt:  # for ctrl + d
          print("\nSee you soon !")
          quit = True
    else:
        print(
            f"{bcolors.FAIL}Input the directory of your images to run the program{bcolors.ENDC}")
        print(
            f"{bcolors.WARNING}usage:\t{bcolors.ENDC}python cli.py <images_directory>")
        return


if __name__ == "__main__":
    main()
