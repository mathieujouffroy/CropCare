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
from tensorflow.keras import callbacks
from prep_data_train import load_split_hdf5, generate_class_weights
from preprocess_tensor import prep_ds_input, create_hf_ds
from custom_callbacks import RocAUCScore
from sklearn.utils.class_weight import compute_class_weight
#import tensorflow_addons as tfa
from transformers import create_optimizer

physical_devices = tf.config.experimental.list_physical_devices('GPU')

logger = logging.getLogger(__name__)


def train_model(args, model_tupl, train_set, valid_set, class_weights):
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
    model = model_tupl[1]

    # Prepare optimizer
    print(f"model --> {model}")
    if model_tupl[0] == "VIT_HF":
        optimizer, lr_schedule = create_optimizer(
        init_lr=args.learning_rate,
        num_train_steps=args.n_training_steps,
        weight_decay_rate=0.01,
        num_warmup_steps=10,
     )
    else:
        if args.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=args.learning_rate, momentum=0.9)
        elif args.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    model.compile(loss=args.loss, optimizer=optimizer, metrics=args.metrics)

    # Define callbacks for debugging and progress tracking
    tb_log_dir = os.path.join(args.model_dir, f'TB_fit')
    checks_path = os.path.join(args.model_dir, 'best-checkpoint-f1')
    callback_lst = [
        callbacks.TensorBoard(histogram_freq=1, log_dir=tb_log_dir),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
        #callbacks.ModelCheckpoint(filepath=checks_path, monitor="val_f1_m",
        #                          save_best_only=True, verbose=1, mode="max"),
        #tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
    ]
    if args.wandb:
        # add WEIGHT INITIALIZATION TRACKING
        wandb_callback = wandb.keras.WandbCallback()
        callback_lst.append(wandb_callback)

    logger.info("\n\n")
    logger.info(f"  =========== TRAINING MODEL {model_tupl[0]} ===========")
    logger.info(f"  Loss = {args.loss}")
    logger.info(f"  Optimizer = {optimizer}")
    logger.info(f"  learning rate = {args.learning_rate}")
    logger.info(f"  Nbr Epochs = {args.n_epochs}")
    logger.info(f"  Nbr of training batch = {args.nbr_train_batch}")
    logger.info(f"  Nbr training steps = {args.n_training_steps}")
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
        #args.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        args.loss = tf.keras.losses.CategoricalCrossentropy()
        args.metrics = [tf.keras.metrics.CategoricalAccuracy(
            name='accuracy', dtype=None)]
        args.metrics = [tf.keras.metrics.SparseCategoricalAccuracy(
            name='accuracy', dtype=None)]


        if args.class_type == 'disease':
            #args.n_classes = 38
            args.n_classes = 3
            args.label_map_path = '../resources/diseases_label_map.json'
        elif args.class_type == 'plants':
            args.n_classes = 14
            args.label_map_path = '../resources/plants_label_map.json'
        else:
            args.n_classes = 14
            args.label_map_path = '../resources/general_diseases_label_map.json'

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


    # Set relevant metrics to evaluate
    #args.metrics.extend([
    ##tf.keras.metrics.CategoricalAccuracy(3, name="top-3-accuracy"),
    #tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name="top-3-accuracy"),
    #tf.keras.metrics.TruePositives(name='tp'),
    #tf.keras.metrics.FalsePositives(name='fp'),
    #tf.keras.metrics.TrueNegatives(name='tn'),
    #tf.keras.metrics.FalseNegatives(name='fn'),
    #f1_m, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
    #tf.keras.metrics.AUC(name='auc'),
    #tf.keras.metrics.AUC(name='prc', curve='PR'),
    #tf.keras.metrics.AUC(name='auc_weighted', label_weights= class_weights),
    ##[tf.keras.metrics.Precision(class_id=i, name=f'precis_{i}') for i in range(4)],
    ##[tf.keras.metrics.Recall(class_id=i, name=f'recall_{i}') for i in range(4)],
    #])

    ## Create Dataset
    if args.transformer:
        train_set = create_hf_ds(args, X_train, y_train)
        valid_set = create_hf_ds(args, X_train, y_train)
    else:
        train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        valid_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))

    train_set = prep_ds_input(args, train_set, 'train')
    valid_set = prep_ds_input(args, valid_set, 'valid')
    print(train_set)
    print(valid_set)
    for elem, label in train_set.take(1):
        img = elem[0].numpy()
        print(f"element shape is {elem.shape}, type is {elem.dtype}")
        print(f"image shape is {img.shape}, type: {img.dtype}")
        print(f"label shape is {label.shape} type: {label.dtype}")

    # Retrieve Models to evaluate
    if args.n_classes == 2:
        models_dict = get_models(args.n_classes-1)
    else:
        models_dict = get_models(args.n_classes, id2label)

    # Set training parameters
    # Nbr training steps is [number of batches] x [number of epochs].
    args.nbr_train_batch = len(list(train_set.as_numpy_iterator()))
    args.n_training_steps = args.nbr_train_batch * args.n_epochs

    logger.info(f"  ---- Training Parameters ----\n\n{args}\n\n")
    logger.info(f"  ***** Running training *****")
    logger.info(f"  train_set = {train_set}")
    logger.info(f"  Nbr of class = {args.n_classes}")
    logger.info(f"  Nbr training examples = {args.len_train}")
    logger.info(f"  Nbr validation examples = {args.len_train}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Class weights = {class_weights}")

    del X_train, X_valid, y_train, y_valid
    gc.collect()

    # Train and evaluate
    for model_d in models_dict.items():
        tf.keras.backend.clear_session()
        # Define directory to save model checkpoints and logs
        date = datetime.datetime.now().strftime("%d:%m:%Y_%H:%M:%S")
        args.model_dir = os.path.join(args.output_dir, f"{model_d[0]}_{date}")
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        if args.wandb:
            dir_name = args.output_dir.split('/')[-1]
            project_name = f"{dir_name }_{args.class_type}-classification"
            cfg = wandb_cfg(args, args.n_training_steps)
            run = wandb.init(project=project_name,
                             job_type="train", name=model_d[0], config=cfg, reinit=True)
            #wandb.init() returns a run object, and you can also access the run object via wandb.run:
            assert run is wandb.run


        trained_model = train_model(args, model_d, train_set, valid_set,
                                   class_weights)
        logger.info("\n")
        logger.info(f"  ***** Evaluating on Validation set *****")
        #compute_training_metrics(args, trained_model, valid_set)
        if args.wandb:
            wandb.run.finish()
            print("\n\n--- FINISH WANDB RUN ---\n")


if __name__ == "__main__":
    main()

### BASELINE
## omit batch norm -≥ then add quickly batch norm
## small regularization at start
## add dropout 2nd run
## early stopping from 1st run
## TEST MODEL TRAINED ON IMAGENET

#tensorboard --logdir logs/fit
