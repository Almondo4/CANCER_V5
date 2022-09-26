
import pandas as pd
import numpy as np


# ## TRAINING Data

train_set = pd.read_csv('../../Data/L1_Enhancement/S_CNV.csv')

labelVector = train_set.iloc[:, -1].values
featureMatrix = train_set.iloc[:, :-1].values


# ## MODEL
# Initilazing the classifier

from sklearn import svm

## MODEL

classifier = svm.SVC(kernel='rbf', random_state = 1,decision_function_shape='ovo', verbose=3, probability=True)


# Results

from Results import calculateResults

calculateResults(classifier,featureMatrix,labelVector)






