import sys
import json
import wandb
import random
import numpy as np
from cli_utils import bcolors, strawb, animate
from dataloader import PlantDataset, store_hdf5, create_transformer_ds, resize_images
from leaf_segmentation import segment_split_set
from sklearn.model_selection import train_test_split

def get_imgs_table(test_x, test_y, nbr_imgs):
    img_dict = dict()
    for img, label in zip(test_x, test_y):
        if label not in img_dict.keys():
            img_dict[label] = []
        img_dict[label].append(img)

    imgs_table = []
    i = 0
    with open("../resources/label_maps/diseases_label_map.json") as f:
        CLASS_INDEX = json.load(f)
    for unique_class, img_lst in img_dict.items():
        sample = random.sample(img_lst, nbr_imgs)
        for img in sample:
            class_name = CLASS_INDEX[str(unique_class)]
            imgs_table.append([i, class_name, wandb.Image(img)])
            i += 1

    return imgs_table


def viz_dataset_wandb(test_x, test_y, nbr_imgs, name):
    imgs_table = get_imgs_table(test_x, test_y, nbr_imgs)
    # Initialize a new W&B run
    run = wandb.init(project='cropdis_vis', reinit=True)
    # Intialize a W&B Artifacts
    ds_artifact = wandb.Artifact(name, type="dataset")

    # create a wandb.Table() with corresponding columns
    columns = ["id", "label", "image"]

    test_table = wandb.Table(data=imgs_table, columns=columns)
    run.log({name: test_table})

    # Add the table to the Artifact
    ds_artifact.add(test_table, name)
    run.log_artifact(ds_artifact)
    wandb.run.finish()


def get_split_sets(seed, class_type, images, labels):
    """
    Prepare the inputs and target split sets for a given classification.

    Args:
        seed(int): seef for the random state
        class_type(str): type of classification (binary, multiclass)
        images(numpy.array): images array
        labels(numpy.array): label array

    Returns:
        images_split_lst(list): list containing the images split sets
        label_split_lst(list): list containing the labels split sets
    """

    # Split train, valid, test
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        images, labels, test_size=0.30, stratify=labels, random_state=seed)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)

    label_split_lst = list([y_train, y_valid, y_test])
    images_split_lst = list([X_train, X_valid, X_test])
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

    return images_split_lst, label_split_lst


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
        raw_images = resize_images(images, (128, 128))
        X_splits, y_splits = get_split_sets(42, label_type, raw_images, labels)
        X_train, X_valid, X_test = X_splits
        y_train, y_valid, y_test = y_splits
        viz_dataset_wandb(X_test, y_test, 5, 'test_ds')
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
                        X_train_seg = segment_split_set(X_train, p_option)
                        X_val_seg = segment_split_set(X_valid, p_option)
                        X_test_seg = segment_split_set(X_test, p_option)
                    else:
                        print('Invalid option')
                        continue

                if options == '2':
                    print('Leag Segmentation HSV mask + dist transform')
                    p_option = input(
                        f"""Chose Image adjustments (brightness, contrast):\n{bcolors.OKBLUE}[0]{bcolors.ENDC} -- No Adjustments\n{bcolors.OKBLUE}[1]{bcolors.ENDC} -- Adjust Contrast\n{bcolors.OKBLUE}[2]{bcolors.ENDC} -- Adjust Lightness and Contrast\n""")
                    if p_option in ['0', '1', '2']:
                        X_train_seg = segment_split_set(X_train, p_option, dist=True)
                        X_val_seg = segment_split_set(X_valid, p_option, dist=True)
                        X_test_seg = segment_split_set(X_test, p_option, dist=True)
                    else:
                        print('Invalid option')
                        continue

                if options in ['1', '2']:
                    dataset_name = f"../resources/datasets/segm_{label_type}_{plant_data.img_nbr}_ds_128.h5"
                    # Get stats from training set for data preprocessing
                    dump_training_stats(X_train_seg, label_type, prefix='segm_')
                    viz_dataset_wandb(X_test_seg, y_test, 5, 'segm_test_ds')
                    store_hdf5(dataset_name,  X_train_seg, X_val_seg, X_test_seg, y_train, y_valid, y_test)

        except EOFError:  # for ctrl + c
          print("\nBye !")

        except KeyboardInterrupt:  # for ctrl + d
          print("\nSee you soon !")

    else:
        print(
            f"{bcolors.FAIL}Input the directory of your images to run the program{bcolors.ENDC}")
        print(
            f"{bcolors.WARNING}usage:\t{bcolors.ENDC}python cli.py <images_directory>")
        return


if __name__ == "__main__":
    main()
