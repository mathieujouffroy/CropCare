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


def plot_roc_curves(args, y_test, y_pred, classes, model_metrics_dir):
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

    if args.wandb:
        rc_name = f"ROC_curves"
        wandb.run.log({rc_name: fig})
    else:
        plt.savefig(f"{model_metrics_dir}/roc_curves.png")

    roc_score = roc_auc_score(ground_truth, predictions, average='weighted')
    return roc_score



def plot_prrc_curves(args, y_test, y_pred, classes, model_metrics_dir):
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

    if args.wandb:
        rc_name = f"Precision_recall_curves"
        wandb.run.log({rc_name: fig})
    else:
        plt.savefig(f"{model_metrics_dir}/precision_recall_curves.png")

@tf.function(jit_compile=True)
def eval_and_pred(model, test_dataset):
    results = model.evaluate(test_dataset)
    y_probs = model.predict(test_dataset)
    y_preds = y_probs.argmax(axis=-1)
    return y_preds

def compute_training_metrics(args, model, m_name, test_dataset):
    """ Compute training metrics for model evaluation. """

    model_metrics_dir = os.path.join(args.output_dir, f"{m_name}_metrics")

    if not os.path.exists(model_metrics_dir):
        os.makedirs(model_metrics_dir)

    with open(args.label_map_path) as f:
        CLASS_INDEX = json.load(f)

    if args.transformer:
        #y_probs = model.predict(test_dataset)
        #y_pred = y_probs.argmax(axis=-1)
        results = eval_and_pred(model, test_dataset)
        f1_sc = None
        roc_score = None
    else:
        y_test = np.concatenate([y for x, y in test_dataset], axis=0)
        x_test = np.concatenate([x for x, y in test_dataset], axis=0)
        results = model.evaluate(x_test, y_test, verbose=0)
        y_probs = model.predict(x_test)

        if args.loss != 'binary_crossentropy':
            y_pred = y_probs.argmax(axis=-1)
            y_test = y_test.argmax(axis=-1)
            truth_label_names = [CLASS_INDEX[str(y)] for y in y_test]
            pred_label_names = [CLASS_INDEX[str(y)] for y in y_pred]
        else:
            y_pred = np.where(y_pred > 0.5, 1,0)
            truth_label_names = y_test
            pred_label_names = y_pred

        accuracy = accuracy_score(y_test, y_pred)
        f1_sc = f1_score(y_test, y_pred, average='weighted')
        matt_score = matthews_corrcoef(y_test, y_pred)
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

        cm = pd.DataFrame(confusion_matrix(truth_label_names, pred_label_names),
                          index=CLASS_INDEX.values(), columns=CLASS_INDEX.values())

        cr = classification_report(
            truth_label_names, pred_label_names, output_dict=True)
        cr_df = pd.DataFrame(cr).transpose()

        fig, ax = plt.subplots(figsize=(16, 20))
        ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.3g')
        ax.set_ylabel('Actual Values ')
        ax.xaxis.set_ticklabels(CLASS_INDEX.values())
        ax.yaxis.set_ticklabels(CLASS_INDEX.values())
        plt.xticks(rotation=90)

        # Display the visualization of the Confusion Matrix.
        if args.wandb:
            columns = ['classes', 'precision', 'recall', 'f1_score', 'support']
            cr_df = cr_df.reset_index()
            cr_wd = wandb.Table(columns=columns, data=cr_df)
            cm_name = f"Confusion_matrix"
            cr_name = f"{m_name}_Classification_report"
            wandb.run.log({cr_name: cr_wd})
            wandb.run.log({cm_name: wandb.Image(fig)})
            wandb.run.log({f"{m_name}_conf_mat_val": wandb.plot.confusion_matrix(
                preds=y_pred, y_true=y_test,
                class_names=list(CLASS_INDEX.values()))})
        else:
            plt.savefig(f"{model_metrics_dir}/confusion_matrix.png")

        roc_score = plot_roc_curves(args, y_test, y_pred, CLASS_INDEX.values(),
                        model_metrics_dir)
        plot_prrc_curves(args, y_test, y_pred, CLASS_INDEX.values(),
                        model_metrics_dir)

        logger.info(f"  ======= METRICS =======")
        logger.info(f"  accuracy = {accuracy}")
        logger.info(f"  f1_score = {f1_sc}")
        logger.info(f"  matthews_corrcoef = {matt_score}")
        logger.info(f"  ROC AUC: {roc_score}")
        logger.info(f"  classification_report:\n\n{cr_df}\n\n")

        ## MODEL INTERPRETABILITY
        #save_and_display_gradcam(args, model, m_name, x_test, y_test, 1, model_metrics_dir)

    return results, f1_sc, roc_score
