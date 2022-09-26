import pandas as pd
import numpy as np


from tensorflow.keras.utils import to_categorical


Hybrid_DATA = pd.read_csv('../../Data/L2_Enhancement\S2_Hybrid_T2.csv')

    # TODO: encode the data first
        # Vector of labels
labelVector =  Hybrid_DATA.iloc[:,-1].values
labelVector = to_categorical(labelVector)
        # featureMatrix
featureMatrix = Hybrid_DATA.iloc[:,:-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(featureMatrix, labelVector, test_size = 0.2, random_state = 0)

# Model
import tensorflow as tf
from tensorflow import keras




# for visualization

import os
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir() # e.g., './my_logs/run_2019_06_07-15_15_22'


# to start from console: tensorboard --logdir ./my_logs
#########################################################################
#########################################################################


#  Model Building
model = keras.models.Sequential()
model.add(keras.layers.Dense(input_shape = [len(featureMatrix[0])], units=1000, activation="relu",
                              name="hiddenL_1"))
model.add(tf.keras.layers.Dropout(rate =0.1))
model.add(keras.layers.Dense( units=500, activation="relu",name="hiddenL_2",kernel_regularizer='l1'))
model.add(tf.keras.layers.Dropout(rate =0.1))
model.add(keras.layers.Dense( units=100, activation="relu",name="hiddenL_3",kernel_regularizer='l1'))
model.add(tf.keras.layers.Dropout(rate =0.1))
model.add(keras.layers.Dense( units=50, activation="relu",name="hiddenL_4",kernel_regularizer='l1'))
model.add(keras.layers.Dense(units=5, activation="softmax", name="outLayer"))

## Compiling

        ### callbacks
# cp = tf.keras.callbacks.ModelCheckpoint("DNN_AndMal.h5",save_best_only =True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        ### Optimizer
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss ="categorical_crossentropy", metrics =["accuracy"])


## Training
history = model.fit(X_train, y_train, batch_size=300, epochs=500,
                    validation_split=0.2,callbacks=[tensorboard_cb,es], use_multiprocessing= True, workers= 16 )

## Testing
# from Results import calculateResults
# calculateResults(model,X_test,y_test)

from sklearn.metrics import classification_report
model_predictions = model.predict(X_test)
# cm = confusion_matrix(labelVector, model_predictions)
print(classification_report(y_test, model_predictions.round(),digits=8))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, model_predictions.round())
print('ROC AUC: %.8f' % auc)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, model_predictions.round())
print('Accuracy: %.8f' % acc)