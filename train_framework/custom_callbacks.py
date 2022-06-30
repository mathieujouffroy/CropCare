import wandb
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback
import numpy as np

class RocAUCScore(Callback):
    def __init__(self, training_data, validation_data):
        self.x = np.concatenate([x for x, y in training_data], axis=0)
        self.y = np.concatenate([y for x, y in training_data], axis=0)
        self.x_val = np.concatenate([x for x, y in validation_data], axis=0)
        self.y_val = np.concatenate([y for x, y in validation_data], axis=0)
        super(RocAUCScore, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        print(y_pred.shape)
        roc = roc_auc_score(self.y, y_pred, average='weighted')
        y_pred_val = self.model.predict(self.x_val)
        print(y_pred_val.shape)
        roc_val = roc_auc_score(self.y_val, y_pred_val, average='weighted')
        print(f'\n  *** ROC AUC Score: {roc} - roc-auc_val: {roc_val} ***')


class ValLogImg(Callback):
  """ Custom callback to log validation images at the end of each training epoch"""

  def __init__(self, validation_set, class_names, num_log_batches=1):
    self.num_batches = num_log_batches
    self.validation_set = validation_set
    self.class_names = class_names


  def on_epoch_end(self, epoch, logs={}):
    # collect validation data and ground truth labels from generator
    VAL_TABLE_NAME = "predictions"

    val_data = np.concatenate([x for x, y in self.validation_set], axis=0)
    val_labels = np.concatenate([y for x, y in self.validation_set], axis=0)
    #val_data, val_labels = np.vstack(val_data), np.vstack(val_labels)

    # generate predictions for the given nbr of validation data batches
    val_probs = self.model.predict(val_data)
    true_ids = val_labels.argmax(axis=1)
    max_preds = val_probs.argmax(axis=1)

    ## log validation predictions alongside the run
    columns=["image", "guess", "truth"]
    for a in self.class_names:
      columns.append("score_" + a)
    predictions_table = wandb.Table(columns = columns)

    ## log image, predicted and actual labels, and all scores
    for img, top_guess, scores, truth in zip(val_data,
                                                       max_preds,
                                                       val_probs,
                                                       true_ids):

      row = [wandb.Image(img), self.class_names[top_guess], self.class_names[truth]]
      for s in scores.tolist():
        row.append(np.round(s, 4))
      predictions_table.add_data(*row)
    wandb.run.log({VAL_TABLE_NAME : predictions_table})
