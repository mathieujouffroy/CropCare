import os
import datetime
import random
import logging
import datetime
import gc
import wandb
import numpy as np
import tensorflow as tf
from metrics import *
from models import *
from utils import *
import math
from tensorflow.keras import callbacks
from prep_data_train import load_split_hdf5, create_hf_ds
from preprocess_tensor import prep_ds_input
from custom_callbacks import RocAUCScore
from custom_loss import poly_loss, poly1_cross_entropy_label_smooth
from sklearn.utils.class_weight import compute_class_weight
#import tensorflow_addons as tfa
from datasets import load_from_disk
from transformers import create_optimizer
from transformers import DefaultDataCollator

physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if len(physical_devices) > 0:
#    tf.config.experimental.set_memory_growth(physical_devices[0], True)

logger = logging.getLogger(__name__)


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



def train_model(args, m_name, model, train_set, valid_set, class_weights):
    """
    Training model on train
    Parameters:
        args():
        model_tupl(tupl):
        train_set(tensorflow.Dataset):
        valid_set(tensorflow.Dataset):
        class_weights:
    Returns:
        model(tensorflow.Model):
    """

    # Prepare optimizer
    if args.transformer:
        lr = 3e-5
        optimizer, lr_schedule = create_optimizer(
        init_lr=lr,
        num_train_steps=args.n_training_steps,
        weight_decay_rate=0.01,
        num_warmup_steps=0,
     )
    else:
        lr = args.learning_rate
        if args.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif args.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(loss=poly_loss, optimizer=optimizer, metrics=args.metrics)

    # Define callbacks for debugging and progress tracking
    checks_path = os.path.join(args.model_dir, 'best-checkpoint-f1')
    callback_lst = [
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
        #callbacks.ModelCheckpoint(filepath=checks_path, monitor="val_f1_m",
        #                          save_best_only=True, verbose=1, mode="max"),
        #tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
    ]
    if args.wandb:
        # add WEIGHT INITIALIZATION TRACKING
        wandb_callback = wandb.keras.WandbCallback()
        callback_lst.append(wandb_callback)
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_f1_m", summary="max")
    else:
        callback_lst.append(callbacks.TensorBoard(histogram_freq=1, log_dir=args.model_dir))

    logger.info("\n\n")
    logger.info(f"  =========== TRAINING MODEL {m_name} ===========")
    logger.info(f"  Loss = {args.loss}")
    logger.info(f"  Optimizer = {optimizer}")
    logger.info(f"  learning rate = {lr}")
    logger.info('\n')

    # Train the model
    model.fit(train_set, epochs=args.n_epochs, validation_data=valid_set,
              class_weight=class_weights,
              callbacks=callback_lst,
              verbose=1)

    return model


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

    # SET LOGGING
    set_logging(args)
    # SET SEED
    set_seed(args)

    # Set relevant loss and accuracy
    if args.class_type == 'healthy':
        args.loss = tf.keras.losses.BinaryCrossentropy()
        args.metrics = [tf.keras.metrics.CategoricalAccuracy(
            name='binary_acc', dtype=None)]
    else:
        # Set relevant loss and metrics to evaluate  
        if args.transformer:
            #tf.keras.mixed_precision.set_global_policy("mixed_float16")
            # one-hot encoded labels because are memory inefficient (GPU memory)
            # guarantee of OOM when you are training a language model with a vast vocabulary size, or big image dataset 
            args.loss = tf.keras.losses.SparseCategoricalCrossentropy()
            args.metrics = [
                tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy', dtype=None),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")
            ]
        else:
            if args.polyloss:
                args.loss = poly1_cross_entropy_label_smooth
            else:
                args.loss = tf.keras.losses.CategoricalCrossentropy()
            args.metrics = [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=None),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top-5-accuracy"),
                f1_m, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC(name='auc'), 
                tf.keras.metrics.AUC(name='prc', curve='PR'),
                #tf.keras.metrics.TruePositives(name='tp'),
                #tf.keras.metrics.FalsePositives(name='fp'),
                #tf.keras.metrics.TrueNegatives(name='tn'),
                #tf.keras.metrics.FalseNegatives(name='fn'),
                #tf.keras.metrics.AUC(name='auc_weighted', label_weights= class_weights),
                ##[tf.keras.metrics.Precision(class_id=i, name=f'precis_{i}') for i in range(4)],
                ##[tf.keras.metrics.Recall(class_id=i, name=f'recall_{i}') for i in range(4)],
            ]

        if args.class_type == 'disease':
            #args.n_classes = 38
            #args.label_map_path = '../resources/label_maps/diseases_label_map.json'
            args.n_classes = 3
            args.label_map_path = '../resources/small_test/diseases_label_map.json'
        elif args.class_type == 'plants':
            args.n_classes = 14
            args.label_map_path = '../resources/label_maps/plants_label_map.json'
        else:
            args.n_classes = 14
            args.label_map_path = '../resources/label_maps/general_diseases_label_map.json'

        with open(args.label_map_path) as f:
            id2label = json.load(f)

        args.class_names = [str(v) for k,v in id2label.items()]
        print(f"  Class names = {args.class_names}")

    # Load the dataset
    X_train, y_train = load_split_hdf5(args.dataset, 'train')
    X_valid, y_valid = load_split_hdf5(args.dataset, 'valid')
    args.len_train = len(X_train)
    args.len_valid = len(X_valid)

    # Set class weights for imbalanced dataset
    if args.class_weights:
        class_weights = generate_class_weights(
            y_train, args.class_type, logger)
    else:
        class_weights = None

    ## Create Dataset
    if args.transformer:
        del X_train, X_valid, y_train, y_valid
        gc.collect()
        train_set = load_from_disk(f'{args.fe_dataset}/train')
        valid_set = load_from_disk(f'{args.fe_dataset}/valid')
        data_collator = DefaultDataCollator(return_tensors="tf")
        print(train_set.features["labels"].names)
        train_set = train_set.to_tf_dataset(
                    columns=['pixel_values'],
                    label_cols=["labels"],
                    shuffle=True,
                    batch_size=32,
                    collate_fn=data_collator)
        valid_set = valid_set.to_tf_dataset(
                    columns=['pixel_values'],
                    label_cols=["labels"],
                    shuffle=True,
                    batch_size=32,
                    collate_fn=data_collator)
    else:
        train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        valid_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        del X_train, X_valid, y_train, y_valid
        gc.collect()

    train_set = prep_ds_input(args, train_set, args.len_train)
    valid_set = prep_ds_input(args, valid_set, args.len_valid)

    for elem, label in train_set.take(1):
        img = elem[0].numpy()
        print(f"element shape is {elem.shape}, type is {elem.dtype}")
        print(f"image shape is {img.shape}, type: {img.dtype}")
        print(f"label shape is {label.shape} type: {label.dtype}")

    # Retrieve Models to evaluate
    if args.n_classes == 2:
        models_dict = get_models(args, args.n_classes-1)
    else:
        models_dict = get_models(args, args.n_classes)

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
    for m_name, model_tupl in models_dict.items():
        model = model_tupl[0]
        mode = model_tupl[1]
        print(f"mode:{mode}")
        print(f"model:{m_name}")
        print(model.summary())
        print(model.inputs)
        print("----------")
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.output_shape, layer.trainable)

        #tf.keras.utils.plot_model(model, show_shapes=True, show_dtype=True)
        #plt.show()
        #print(tf.keras.utils.plot_model(model, show_shapes=True, show_dtype=True))
        #plt.show()
        #print("----------")
        #for layer in model.layers:
        #    if 'input' in layer.__class__.__name__:
        #        print(layer.name, layer.input_shape)
        #    if "InputLayer" == layer.__class__.__name__:
        #        print(f"in normal layers : {layer.name}")
        #    if "Functional" == layer.__class__.__name__:
        #        for _l in layer.layers:
        #            print(_l.name)

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

        trained_model = train_model(args, m_name, model, train_set, valid_set, class_weights)
        
        if args.eval_during_training:
            X_test, y_test = load_split_hdf5(args.dataset, 'test')

            # Set parameters
            args.len_test = len(X_test)
            args.nbr_test_batch = int(math.ceil(args.len_test / args.batch_size))

            if args.transformer:
                test_set = load_from_disk(f'{args.fe_dataset}/test')
                data_collator = DefaultDataCollator(return_tensors="tf")
                print(test_set.features["labels"].names)
                test_set = test_set.to_tf_dataset(
                            columns=['pixel_values'],
                            label_cols=["labels"],
                            shuffle=True,
                            batch_size=32,
                            collate_fn=data_collator)
            else:
                test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))

            test_set = prep_ds_input(args, test_set, args.len_test)
            logger.info("\n")
            logger.info(f"  ***** Evaluating on Validation set *****")
            compute_training_metrics(args, trained_model, m_name, mode, test_set)

        if args.wandb:
            wandb.run.finish()
            print("\n\n--- FINISH WANDB RUN ---\n")

if __name__ == "__main__":
    main()
