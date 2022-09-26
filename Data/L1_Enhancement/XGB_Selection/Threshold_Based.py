

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate

def calculateResults2 (optimalModel, featureMatrix,labelVector):
    results = cross_validate(optimalModel, featureMatrix, labelVector, scoring=('accuracy', 'precision_weighted',
                                                                                'f1_weighted', 'recall_weighted',
                                                                                'roc_auc_ovo_weighted'), cv=10,
                             verbose=0, n_jobs=-1, return_estimator=True)

    return results

# ## TRAINING Data

train_set = pd.read_csv('../../Baseline/B_DNA.csv')

labelVector = train_set.iloc[:, -1].values
featureMatrix = train_set.iloc[:, :-1].values


# Extracting feaaure Importance

# ## MODEL

import xgboost as xgb

# XGB DATA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(featureMatrix, labelVector, test_size = 0.2, random_state = 0)
train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)

param = {
    "max_depth":1000,
    "eta": 0.3,
    # "objective": "mutli:softmax",
    "num_class": 5,
    "verbosity": 0}
epochs = 20
model = xgb.XGBClassifier(**param)

# initial results inspection
from Results import calculateResults

calculateResults(model,featureMatrix,labelVector)

# Feature importance time
model.fit(featureMatrix,labelVector)

thresholds = np.sort(model.feature_importances_)

i = thresholds.mean()






######  Threshold tests

from xgboost import plot_importance
from matplotlib import pyplot
plot_importance(model)
pyplot.show()

thresholds_new = np.sort([ x for x in thresholds if x != 0.0 ])
thresholds_new= thresholds_new[::-1]

from sklearn.feature_selection import SelectFromModel

modelth = xgb.XGBClassifier({"max_depth": 1000,
                             "eta": 0.3,
                             # "objective": "mutli:softmax",
                             "num_class": 5,
                             "verbosity": 0})
options =[]
for thresh in thresholds_new:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(featureMatrix)
    # train model
    modelth.fit(select_X_train, labelVector) # <---

    # eval model
    select_X_test = selection.transform(featureMatrix)
    results= calculateResults2(modelth, featureMatrix, labelVector)

    r = str("Thresh=%.3f, n=%d, Accuracy: %.2f%%%%, Presicision: %.2f%%%%, Recall: %.2f%%%%, F1: %.2f%%%%, ROC_AUC: %.2f%%%%, "
          % (thresh, select_X_train.shape[1],results['test_accuracy'].mean() * 100,
             results['test_precision_weighted'].mean() * 100,
             results['test_recall_weighted'].mean() * 100,
             results['test_f1_weighted'].mean() * 100,
             results['test_roc_auc_ovo_weighted'].mean() * 100))
    print(r)
    options.append(r)




with open("Thresholds.txt", "w") as txt_file:
    for line in options:
        txt_file.write(" ".join(line) + "\n")




# ## Training & TESTING\
from sklearn.model_selection import cross_validate
results = cross_validate(model, featureMatrix, labelVector, scoring=('accuracy', 'precision_weighted',
                                                                            'f1_weighted', 'recall_weighted',
                                                                            'roc_auc_ovo_weighted'), cv=10,
                         verbose=0, n_jobs=-1, return_estimator=True)






