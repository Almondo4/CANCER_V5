
import pandas as pd
import numpy as np


# ## TRAINING Data

train_set = pd.read_csv('../../Data/L2_Enhancement\Data/S2_Hybrid_T1.csv')

labelVector = train_set.iloc[:, -1].values
featureMatrix = train_set.iloc[:, :-1].values



# ## MODEL

# Initializing
import xgboost as xgb

# XGB DATA
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(featureMatrix, labelVector, test_size = 0.2, random_state = 0)
# train = xgb.DMatrix(X_train, label=y_train)
# test = xgb.DMatrix(X_test, label=y_test)

param = {
    "max_depth":100,
    "eta": 0.6,
    # "objective": "mutli:softmax",
    "num_class": 5,
    "verbosity": 3}
epochs = 200
model = xgb.XGBClassifier(**param)

# # Training
# model.fit(X_train, y_train)
#


# ## TESTING

# Results

from Results import calculateResults

calculateResults(model,featureMatrix,labelVector)






