
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


# ## TRAINING Data

train_set = pd.read_csv('./unclear.csv')

labelVector = train_set.iloc[:, -1].values
featureMatrix = train_set.iloc[:, :-1].values





# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(featureMatrix)
fitted = pca.fit_transform(featureMatrix)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)



import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(projection='3d')
# ax = fig.add_subplot(projection='3d')
ax.scatter(fitted[:,0], fitted[:,1], fitted[:,2], c=labelVector)
plt.show()