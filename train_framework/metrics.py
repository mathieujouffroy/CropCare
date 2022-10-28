import os
import wandb
import json
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score,  accuracy_score,  matthews_corrcoef
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer
from train_framework.utils import logging
from train_framework.interpretability import *

logger = logging.getLogger(__name__)


def recall_m(y_true, y_pred):
    """
    Recall metric. Only computes a batch-wise average of recall.
    Recall is a metric for multi-label classification which indicates
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """
    Precision metric. Only computes a batch-wise average of precision.
    Precision is metric for multi-label classification which indicates
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    """ F1 Score """

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def matt_coeff(y_true, y_pred):
    """ Matthews correlation coefficient. """

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())


def plot_roc_curves(args, y_test, y_pred, classes, model_metrics_dir, m_type):
    """ Plots the ROC curves for our classes. """

    fig, c_ax = plt.subplots(1, 1, figsize=(18, 18))

    # Transform problem in One vs All
    # ground_truth is -> (n_img, n_class)
    # predictions is -> (n_img)
    lb = LabelBinarizer()

    # Passing a 2D matrix for multilabel classification
    lb.fit(y_test)
    ground_truth = lb.transform(y_test)
    predictions = lb.transform(y_pred)

    # Iterate over each class
    for id, c_label in enumerate(classes):
        fpr, tpr, th = roc_curve(
            ground_truth[:, id].astype(int), predictions[:, id])
        c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label='Random Guessing')
    c_ax.legend(loc='lower right')
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.savefig(f"{model_metrics_dir}/roc_curves.png")

    if args.wandb:
        if m_type == "eval":
            rc_name = "ROC_CURVES_VAL"
        else:
            rc_name = "ROC_CURVES_TEST"
        wandb.run.log({rc_name: plt})

    roc_score = roc_auc_score(ground_truth, predictions, average='weighted')
    logger.info(f"ROC AUC: {roc_score}")


def plot_prrc_curves(args, y_test, y_pred, classes, model_metrics_dir, m_type):
    """ Plots the precision and recall curves for our classes. """

    fig, c_ax = plt.subplots(1, 1, figsize=(18, 18))
    # Transform problem in One vs All
    lb = LabelBinarizer()
    # Passing a 2D matrix for multilabel classification
    lb.fit(y_test)
    ground_truth = lb.transform(y_test)
    predictions = lb.transform(y_pred)
    # Iterate over each class
    for id, c_label in enumerate(classes):
        prec, recall, th = precision_recall_curve(
            ground_truth[:, id].astype(int), predictions[:, id])
        c_ax.plot(prec, recall, label=c_label)
    c_ax.legend(loc='lower right')
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.savefig(f"{model_metrics_dir}/precision_recall_curves.png")

    if args.wandb:
        if m_type == "eval":
            rc_name = "PREC-RECALL_VAL"
        else:
            rc_name = "PREC-RECALL_TEST"
        wandb.run.log({rc_name: plt})


def compute_training_metrics(args, model, m_name, test_dataset, m_type='train'):
    """ Compute training metrics for model evaluation. """

    model_metrics_dir = os.path.join(args.output_dir, f"{m_name}_metrics")

    if not os.path.exists(model_metrics_dir):
        os.makedirs(model_metrics_dir)

    y_test = np.concatenate([y for x, y in test_dataset], axis=0)
    x_test = np.concatenate([x for x, y in test_dataset], axis=0)

    with open(args.label_map_path) as f:
        CLASS_INDEX = json.load(f)


    results = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"  Result of evaluation:")
    logger.info(f"  {results}")
    logger.info(f"  loss: {results[0]}")
    logger.info(f"  acc: {results[1]}")


    y_probs = model.predict(x_test)
    y_pred = y_probs.argmax(axis=-1)

    logger.info(f"  Shape of y_pred:{y_pred.shape}")

    #for img, label in zip(x_test, y_test):
    #    label = label.argmax(axis=-1)
    #    yyy = model.predict(x_test)
    #    y_ppp = yyy.argmax(axis=-1)
    #    print(label)
    #    print(y_ppp)
    #    truth_label_names = CLASS_INDEX[str(label)]
    #    pred_label_names = CLASS_INDEX[str(label)]
    #    print(truth_label_names)
    #    print(pred_label_names)

    if args.loss != 'binary_crossentropy':
        y_test = y_test.argmax(axis=-1)
        truth_label_names = [CLASS_INDEX[str(y)] for y in y_test]
        pred_label_names = [CLASS_INDEX[str(y)] for y in y_pred]

    print(y_pred.shape)
    print(y_test.shape)

    cm = pd.DataFrame(confusion_matrix(y_test, y_pred),
                      index=CLASS_INDEX.values(), columns=CLASS_INDEX.values())

    cr = classification_report(
        truth_label_names, pred_label_names, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()

    accuracy = accuracy_score(y_test, y_pred)
    f1_sc = f1_score(y_test, y_pred, average='weighted')
    matt_score = matthews_corrcoef(y_test, y_pred)
    logger.info(f"  ======= METRICS =======")
    logger.info(f"  accuracy = {accuracy}")
    logger.info(f"  f1_score = {f1_sc}")
    logger.info(f"  matthews_corrcoef = {matt_score}")
    logger.info(f"  classification_report:\n\n{cr_df}\n\n")
    logger.info(f"  confusion matrix:\n\n{cm}\n\n")

    fig, ax = plt.subplots(figsize=(16, 20))
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.3g')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(CLASS_INDEX.values())
    ax.yaxis.set_ticklabels(CLASS_INDEX.values())
    plt.xticks(rotation=90)
    #plt.show()
    #plt.savefig(f"{model_metrics_dir}/confusion_matrix.png")

    # Display the visualization of the Confusion Matrix.
    if args.wandb:
        columns = ['classes', 'precision', 'f1_score', 'support']
        cr_df = cr_df.reset_index()
        cr_wd = wandb.Table(columns=columns, data=cr_df)
        if m_type == 'eval':
            cm_name = "CONFUSION_MATRIX_TEST"
            cr_name = "CLASSIFICATION_REPORT_TEST"
        else:
            cm_name = "CONFUSION_MATRIX_VAL"
            cr_name = "CLASSIFICATION_REPORT_VALID"
        wandb.run.log({cr_name: cr_wd})
        wandb.run.log({cm_name: wandb.Image(fig)})
        wandb.run.log({"conf_mat_val": wandb.plot.confusion_matrix(
            preds=y_pred, y_true=y_test,
            class_names=list(CLASS_INDEX.values()))})

    #plot_roc_curves(args, y_test, y_pred, CLASS_INDEX.values(),
    #                model_metrics_dir, m_type)
#
    #plot_prrc_curves(args, y_test, y_pred, CLASS_INDEX.values(),
    #                model_metrics_dir, m_type)
#
    ## MODEL INTERPRETABILITY
    save_and_display_gradcam(args, model, m_name, x_test, y_test, 1, model_metrics_dir)
