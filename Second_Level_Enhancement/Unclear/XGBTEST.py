
import pandas as pd
import numpy as np


# ## TRAINING Data

train_set = pd.read_csv('L2_Currated.csv')

labelVector = train_set.iloc[:, -1].values
featureMatrix = train_set.iloc[:, :-1].values



# ## MODEL

# Initializing
import xgboost as xgb

param = {
    "max_depth":100,
    "eta": 0.6,
    # "objective": "mutli:softmax",
    "num_class": 4,
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






