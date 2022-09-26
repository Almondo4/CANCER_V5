
import pandas as pd
import numpy as np


# ## TRAINING Data

train_set = pd.read_csv('../../Data/Baseline/B_HYBRID.csv')

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




import shap

shap.initjs()

# explain the model's predictions using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(featureMatrix)

# visualize the first prediction's explanation

features_list = list(train_set.columns)

shap.summary_plot(shap_values[1], featureMatrix, feature_names = features_list, max_display=20,plot_size= [12,12], show = False)
shap.summary_plot(shap_values, featureMatrix, plot_type="bar", class_names= [0,1,2,3,4], feature_names = features_list)

hmexp = shap.Explainer(model, featureMatrix)
hmsv = hmexp(featureMatrix[:606])
shap.plots.heatmap(shap_values, feature_values=hmsv.abs.max(0))

# shap_values = shap.TreeExplainer(model).shap_values(featureMatrix)
# shap.summary_plot(shap_values, X_train, plot_type="bar")

# https://medium.com/mlearning-ai/shap-force-plots-for-classification-d30be430e195





# XGBoost Feature importance

sorted_idx = model.feature_importances_.argsort()
plt.barh(features_list[sorted_idx], model.feature_importances_[sorted_idx])
plt.xlabel("Xgboost Feature Importance")
