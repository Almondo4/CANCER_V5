
import pandas as pd
import numpy as np


# ## TRAINING Data

train_set = pd.read_csv('../../Data/L1_Enhancement/S_CNV.csv')

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

top50.to_csv('top50_L1_CNV.csv',index=False)



#  Heatmap

# from matplotlib import pyplot as plt
# import seaborn as sns
# def correlation_heatmap(train):
#     correlations = train.corr()
#
#     fig, ax = plt.subplots(figsize=(10, 10))
#     sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', cmap="YlGnBu",
#                 square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70}
#                 )
#     plt.show();
#
#
# correlation_heatmap(X_train[train_set.columns[model.feature_importances_.argsort()]])


# SHAP FEATURE IMPORTANCE


import shap

shap.initjs()

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(featureMatrix)

# visualize the first prediction's explanation

shap.summary_plot(shap_values[1], featureMatrix, feature_names = features_list, max_display=20,plot_size= [12,12], show = False)
shap.summary_plot(shap_values, featureMatrix, plot_type="bar", class_names= [0,1,2,3,4], feature_names = features_list)

hmexp = shap.Explainer(model, featureMatrix)
hmsv = hmexp(featureMatrix[:606])
shap.plots.heatmap(shap_values, feature_values=hmsv.abs.max(0))

# shap_values = shap.TreeExplainer(model).shap_values(featureMatrix)
# shap.summary_plot(shap_values, X_train, plot_type="bar")

