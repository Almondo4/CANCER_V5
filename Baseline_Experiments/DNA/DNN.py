import pandas as pd
import numpy as np


from tensorflow.keras.utils import to_categorical


DNA_DATA = pd.read_csv('../../Data/Baseline/B_DNA.csv')

    # TODO: encode the data first
        # Vector of labels
labelVector =  DNA_DATA.iloc[:,-1].values
labelVector = to_categorical(labelVector)
        # featureMatrix
featureMatrix = DNA_DATA.iloc[:,:-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(featureMatrix, labelVector, test_size = 0.3, random_state = 0)

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
model.add(keras.layers.Dense(input_shape = [len(featureMatrix[0])], units=700, activation="relu",
                              name="hiddenL_1"))
model.add(tf.keras.layers.Dropout(rate =0.25))
model.add(keras.layers.Dense(units=100, activation="relu", kernel_regularizer='l1', name="hiddenL_2"))
# model.add(tf.keras.layers.Dropout(rate =0.2))
# model.add(keras.layers.Dense(units=500, activation="relu", kernel_regularizer='l2', name="hiddenL_3"))
# model.add(tf.keras.layers.Dropout(rate =0.15))
model.add(keras.layers.Dense(units=5, activation="softmax", name="outLayer"))

## Compiling

        ### callbacks
# cp = tf.keras.callbacks.ModelCheckpoint("DNN_AndMal.h5",save_best_only =True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
        ### Optimizer
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss ="categorical_crossentropy", metrics =["accuracy"])


## Training
history = model.fit(X_train, y_train, batch_size=2046, epochs=500,
                    validation_split=0.2,callbacks=[tensorboard_cb])

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




# Corss Val CAN BE USED TO CONFIRM THE UNOVERFITTD DNN
# def create_baseline():
#     # create model
#     model = Sequential()
#     model.add(Dense(60, input_dim=11, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5,     verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



