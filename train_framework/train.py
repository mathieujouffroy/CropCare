import os
import logging
import datetime
import wandb
import numpy as np
import tensorflow as tf
from transformers import create_optimizer, AdamWeightDecay, get_scheduler
from sklearn.utils.class_weight import compute_class_weight
from train_framework.custom_callbacks import RocAUCScore
from train_framework.utils import logging

physical_devices = tf.config.experimental.list_physical_devices('GPU')
logger = logging.getLogger(__name__)

def generate_class_weights(y_train, class_type):
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
    Compiles and fits the model.

    Parameters:
        args: Argument Parser
        m_name: Model name
        model: Model to train 
        train_set(tensorflow.Dataset): training set 
        valid_set(tensorflow.Dataset): validation set
        class_weights: Weights for imbalanced classification
    Returns:
        model(tensorflow.Model): trained model
    """

    # Prepare optimizer
    if args.transformer:
        #--model vit_small_patch16_224 --sched cosine --epochs 300 --opt adamp -j 8 --warmup-lr 1e-6 --mixup .2 --model-ema --model-ema-decay 0.99996 --aa rand-m9-mstd0.5-inc1 --remode pixel --reprob 0.25 --amp --lr .001 --weight-decay .01 -b 256
        # cosine scheduler
        #lr_scheduler_type="cosine" -≥ VIT
        #optimizer = AdamWeightDecay(lr=2e-5, weight_decay_rate=0.01)
        #lr_scheduler = get_scheduler(
        #    "linear",
        #    optimizer=optimizer,
        #    num_warmup_steps=0,
        #    num_training_steps=args.n_training_steps,
        #)
        lr = 2e-5
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

    model.compile(loss=args.loss, optimizer=optimizer, metrics=args.metrics)

    # Define callbacks for debugging and progress tracking
    checks_path = os.path.join(args.model_dir, 'best-checkpoint-f1')
    callback_lst = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.05, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=checks_path, monitor="val_f1_m", save_best_only=True, verbose=1, mode="max"),
        #tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
    ]
    if args.wandb:
        wandb_callback = wandb.keras.WandbCallback()
        callback_lst.append(wandb_callback)
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_f1_m", summary="max")
    else:
        callback_lst.append(tf.keras.callbacks.TensorBoard(histogram_freq=1, log_dir=args.model_dir))

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