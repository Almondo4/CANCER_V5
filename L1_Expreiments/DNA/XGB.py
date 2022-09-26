
import pandas as pd
import numpy as np


# ## TRAINING Data

train_set = pd.read_csv('../../Data/L1_Enhancement/S_DNA.csv')

labelVector = train_set.iloc[:, -1].values
featureMatrix = train_set.iloc[:, :-1].values



# ## MODEL

# Initializing
import xgboost as xgb


param = {
    "max_depth":1000,
    "eta": 0.3,
    # "objective": "mutli:softmax",
    "num_class": 5,
    "verbosity": 3}
epochs = 20
model = xgb.XGBClassifier(**param)

# # Training
# model.fit(X_train, y_train)
#


# ## TESTING

# Results

from Results import calculateResults

calculateResults(model,featureMatrix,labelVector)



# Getting the feature importance

# getting the features

features_list = list(train_set.columns)
features_list.remove('class')

# fitting the model

# XGB DATA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(featureMatrix, labelVector, test_size = 0.2, random_state = 0)
train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)

model.fit(X_train, y_train)

# getting the top 50
results=pd.DataFrame()
results['columns']=features_list
results['importances'] = model.feature_importances_
results.sort_values(by='importances',ascending=False,inplace=True)

top50 = results[:50]

top50.to_csv('top50_L1_DNA.csv',index=False)


