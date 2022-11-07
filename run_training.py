import os
import datetime
import logging
import datetime
import gc
import wandb
import tensorflow as tf
import math
import json
from datasets import load_from_disk
from transformers import DefaultDataCollator
from train_framework.metrics import compute_training_metrics, f1_m
from train_framework.models import get_models
from train_framework.utils import set_logging, set_seed, set_wandb_project_run, parse_args
from train_framework.prep_data_train import load_split_hdf5
from train_framework.preprocess_tensor import prep_ds_input
from train_framework.custom_loss import poly_loss, poly1_cross_entropy_label_smooth
from train_framework.train import generate_class_weights, train_model

logger = logging.getLogger(__name__)


def main():

    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    elif (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    # set logging
    set_logging(args)
    # set seed
    set_seed(args)

    # Set relevant loss and accuracy
    if args.class_type == 'healthy':
        args.n_classes = 2
        args.class_names = ['healthy', 'not_healthy']
        args.loss = tf.keras.losses.BinaryCrossentropy()
        args.metrics = [tf.keras.metrics.CategoricalAccuracy(
            name='binary_acc', dtype=None)]
    else:
        # Set relevant loss and metrics to evaluate
        if args.transformer:
            args.n_epochs = 6
            args.learning_rate = 2e-5
            args.loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True)
            # one-hot encoded labels because are memory inefficient (GPU memory)
            # guarantee of OOM when you are training a language model with a vast vocabulary size, or big image dataset
            args.metrics = [
                tf.keras.metrics.SparseCategoricalAccuracy(
                    name='accuracy', dtype=None),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(
                    k=5, name="top-5-accuracy")
            ]
        else:
            if args.polyloss:
                args.loss = poly1_cross_entropy_label_smooth
            else:
                args.loss = tf.keras.losses.CategoricalCrossentropy()

            args.metrics = [
                tf.keras.metrics.CategoricalAccuracy(
                    name='accuracy', dtype=None),
                tf.keras.metrics.TopKCategoricalAccuracy(
                    k=5, name="top-5-accuracy"),
                f1_m, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.AUC(name='prc', curve='PR'),
                #tf.keras.metrics.AUC(name='auc_weighted', label_weights= class_weights),
            ]

        if args.class_type == 'disease':
            args.n_classes = 38
            args.label_map_path = 'resources/label_maps/diseases_label_map.json'
        elif args.class_type == 'plants':
            args.n_classes = 14
            args.label_map_path = 'resources/label_maps/plants_label_map.json'
        else:
            args.n_classes = 14
            args.label_map_path = 'resources/label_maps/general_diseases_label_map.json'

        with open(args.label_map_path) as f:
            id2label = json.load(f)

        args.class_names = [str(v) for k, v in id2label.items()]

    logger.info(f"  Class names = {args.class_names}")

    ## Create Dataset
    if args.transformer:
        args.input_shape = [224, 224, 3]
        img_size = (args.input_shape[0:2])

        if args.feature_extractor == "vit":
            ds_path = "../block_storage/transformers/vit"
        elif args.feature_extractor == "swin":
            ds_path = "../block_storage/transformers/swin"
        elif args.feature_extractor == "convnext":
            ds_path = "../block_storage/transformers/convnext"

        train_set = load_from_disk(f'{ds_path}/train')
        valid_set = load_from_disk(f'{ds_path}/valid')
        data_collator = DefaultDataCollator(return_tensors="tf")

        y_train = train_set['labels']
        args.len_train = train_set.num_rows
        args.len_valid = valid_set.num_rows

        train_set = train_set.to_tf_dataset(
            columns=['pixel_values'],
            label_cols=["labels"],
            shuffle=True,
            batch_size=32,
            collate_fn=data_collator
        )
        valid_set = valid_set.to_tf_dataset(
            columns=['pixel_values'],
            label_cols=["labels"],
            shuffle=True,
            batch_size=32,
            collate_fn=data_collator
        )

    else:
        # Load the dataset
        assert os.path.isfile(args.dataset)
        img_size = args.input_shape[0:2]
        X_train, y_train = load_split_hdf5(args.dataset, 'train')
        X_valid, y_valid = load_split_hdf5(args.dataset, 'valid')
        train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        valid_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        args.len_train = len(X_train)
        args.len_valid = len(X_valid)
        del X_train, X_valid, y_valid
        gc.collect()

    # Set class weights for imbalanced dataset
    if args.class_weights:
        class_weights = generate_class_weights(y_train, args.class_type)
    else:
        class_weights = None
    del y_train
    gc.collect()

    train_set = prep_ds_input(args, train_set, args.len_train, img_size)
    valid_set = prep_ds_input(args, valid_set, args.len_valid, img_size)

    for elem, label in train_set.take(1):
        img = elem[0].numpy()
        logger.info(f"batch shape is {elem.shape}, type is {elem.dtype}")
        logger.info(f"image shape is {img.shape}, type: {img.dtype}")
        logger.info(f"label shape is {label.shape} type: {label.dtype}")

    # Retrieve Models to evaluate
    if args.n_classes == 2:
        args.n_classes = 1
        models_dict = get_models(args)
    else:
        models_dict = get_models(args)

    # Set training parameters
    args.nbr_train_batch = int(math.ceil(args.len_train / args.batch_size))
    # Nbr training steps is [number of batches] x [number of epochs].
    args.n_training_steps = args.nbr_train_batch * args.n_epochs

    logger.info(f"  ---- Training Parameters ----\n\n{args}\n\n")
    logger.info(f"  ***** Running training *****")
    logger.info(f"  train_set = {train_set}")
    logger.info(f"  Nbr of class = {args.n_classes}")
    logger.info(f"  Nbr training examples = {args.len_train}")
    logger.info(f"  Nbr validation examples = {args.len_valid}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Nbr Epochs = {args.n_epochs}")
    logger.info(f"  Nbr of training batch = {args.nbr_train_batch}")
    logger.info(f"  Nbr training steps = {args.n_training_steps}")
    logger.info(f"  Class weights = {class_weights}")

    # Train and evaluate
    for m_name, model in models_dict.items():
        tf.keras.backend.clear_session()
        # Define directory to save model checkpoints and logs
        date = datetime.datetime.now().strftime("%d:%m:%Y_%H:%M:%S")
        if args.polyloss:
            m_name = m_name+"_poly"

        args.model_dir = os.path.join(args.output_dir, f"{m_name}_{date}")
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

        # define wandb run and project
        if args.wandb:
            set_wandb_project_run(args, m_name)

        trained_model = train_model(
            args, m_name, model, train_set, valid_set, class_weights)

        if args.eval_during_training:
            if args.transformer:
                args.input_shape = (3, 224, 224)
                test_set = load_from_disk(f'{ds_path}/test')
                args.len_test = test_set.num_rows
                data_collator = DefaultDataCollator(return_tensors="tf")
                test_set = test_set.to_tf_dataset(
                    columns=['pixel_values'],
                    label_cols=["labels"],
                    shuffle=True,
                    batch_size=32,
                    collate_fn=data_collator)
            else:
                X_test, y_test = load_split_hdf5(ds_path, 'test')
                args.len_test = len(X_test)
                test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))
                del X_test, y_test
                gc.collect()
            test_set = prep_ds_input(args, test_set, args.len_test, img_size)
            logger.info(f"\n  ***** Evaluating on Test set *****")
            compute_training_metrics(args, trained_model, m_name, test_set)

        if args.wandb:
            wandb.run.finish()


if __name__ == "__main__":
    main()
