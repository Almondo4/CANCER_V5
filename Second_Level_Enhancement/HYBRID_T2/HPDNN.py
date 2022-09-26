import pandas as pd
import numpy as np


from tensorflow.keras.utils import to_categorical


Hybrid_DATA = pd.read_csv('Hybrid_RFE.csv')

 # TODO: encode the data first
        # Vector of labels
labelVector =  Hybrid_DATA.iloc[:,-1].values
labelVector = to_categorical(labelVector)
        # featureMatrix
featureMatrix = Hybrid_DATA.iloc[:,:-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(featureMatrix, labelVector, test_size = 0.3, random_state = 0)

# Model
import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp

# for visualization

import os
root_logdir = os.path.join(os.curdir, "../my_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir() # e.g., './my_logs/run_2019_06_07-15_15_22'
# to start from console: tensorboard --logdir ./my_logs


# preparing the HParam search
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([50,100, 500, 900]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )
#  Model Building

def train_test_model(hparams):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_shape = [len(featureMatrix[0])], units=hparams[HP_NUM_UNITS], activation="relu",
                              name="hiddenL_1"),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(units=hparams[HP_NUM_UNITS], activation="relu", kernel_regularizer='l1', name="hiddenL_2"),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(units=5, activation="softmax", name="outLayer"),
  ])
  model.compile(
      optimizer=hparams[HP_OPTIMIZER],
      loss='categorical_crossentropy',
      metrics=['accuracy'],
  )
  model.fit(X_train, y_train, batch_size=2048, epochs=500,
                      validation_split=0.2,)
  _, accuracy = model.evaluate(X_test, y_test)
  return accuracy
# model = keras.models.Sequential()
# model.add(keras.layers.Dense(input_shape = [len(featureMatrix[0])], units=500, activation="relu",
#                               name="hiddenL_1"))
# model.add(tf.keras.layers.Dropout(rate =0.25))
# model.add(keras.layers.Dense(units=100, activation="relu", kernel_regularizer='l1', name="hiddenL_2"))
# model.add(tf.keras.layers.Dropout(rate =0.25))
# model.add(keras.layers.Dense(units=100, activation="relu", kernel_regularizer='l2', name="hiddenL_3"))
# # model.add(tf.keras.layers.Dropout(rate =0.15))
# model.add(keras.layers.Dense(units=5, activation="softmax", name="outLayer"))



## Compiling


def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)




        ### callbacks
# cp = tf.keras.callbacks.ModelCheckpoint("DNN_AndMal.h5",save_best_only =True)
# tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
        ### Optimizer
# opt = tf.keras.optimizers.Adam()
# model.compile(optimizer=opt, loss ="categorical_crossentropy", metrics =["accuracy"])


## Training
# history = model.fit(X_train, y_train, batch_size=2048, epochs=500,
#                     validation_split=0.2,callbacks=[tensorboard_cb])


# RUN
session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for optimizer in HP_OPTIMIZER.domain.values:
      hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning/' + run_name, hparams)
      session_num += 1
# TO confirm results later

# from sklearn.metrics import classification_report
# model_predictions = model.predict(X_test)
# # cm = confusion_matrix(labelVector, model_predictions)
# print(classification_report(y_test, model_predictions.round(),digits=4))
#
# from sklearn.metrics import roc_auc_score
# auc = roc_auc_score(y_test, model_predictions.round())
# print('ROC AUC: %f' % auc)