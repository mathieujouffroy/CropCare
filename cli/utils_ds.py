def split_for_training(i, class_img_paths, train_path, val_path, test_path, train_y, val_y, test_y, training_size=0.7, validation_size=0.15, test_size=0.15):
    """
    Splits the dataset into training, validation and test sets.

    Args:
        dat_set_path (bytes): path to the dataset
        class_img_paths (list): list of paths to the images per class
        healthy_set (list): list of healthy/sick labels per class
        plant_set (list): list of plant labels per class
        disease_set (list): list of disease labels per class
    """
    n_imgs_per_class = len(class_img_paths)
    train_path.extend(class_img_paths[:int(n_imgs_per_class * training_size)])
    train_y.extend([i] * int(n_imgs_per_class * training_size))

    val_path.extend(class_img_paths[int(n_imgs_per_class * training_size)
                    :int(n_imgs_per_class * (training_size + validation_size))])
    val_y.extend([i] * len(class_img_paths[int(n_imgs_per_class * training_size)
                 :int(n_imgs_per_class * (training_size+validation_size))]))

    test_path.extend(
        class_img_paths[int(n_imgs_per_class * (training_size + validation_size)):])
    test_y.extend(
        [i] * len(class_img_paths[int(n_imgs_per_class * (training_size + validation_size)):]))
    return train_path, val_path, test_path, train_y, val_y, test_y
