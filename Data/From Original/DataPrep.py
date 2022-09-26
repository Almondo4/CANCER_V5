import pandas as pd
import numpy as np


# DNA

DNA_DATA = pd.read_csv('../Data/Original/brca_methDNA_subtype.csv')

    # TODO: encode the data first
        # Vector of labels
# transposing
DNA_DATA_Trans = DNA_DATA.T
labelVector =  DNA_DATA_Trans.index.values
# removing "features"
labelVector= np.delete(labelVector,0)

# extracting the feature names:
DNA_features = list(DNA_DATA['Features'])
DNA_features.append('Class')

# Encoding Classes
import re
regexpA = re.compile(r'lminalA')
regexpB = re.compile(r'lminalB')
regexpT = re.compile(r'TNBC')
regexpH = re.compile(r'ERBB')
regexpU = re.compile(r'normal')
def labelEncoding(arr):
    for i, e in enumerate(arr):
        if regexpA.search(e):
            arr[i]= '0'
        elif regexpB.search(e):
            arr[i]= '1'
        elif regexpT.search(e):
            arr[i]= '2'
        elif regexpH.search(e):
            arr[i]= '3'
        elif regexpU.search(e):
            arr[i]= '4'

    print(arr)
    return arr

labels = labelEncoding(labelVector)
        # featureMatrix
featureMatrix = DNA_DATA_Trans.iloc[1:,:].values

# Fusing the Hybrid matrix with the feature label
DNA_DATA_New = np.insert(featureMatrix, 14285, labels, axis=1)

# Transforming into a dataFrame
DNA_DATA_New = pd.DataFrame(DNA_DATA_New)
# Adding the name of features:
DNA_DATA_New.columns = DNA_features

# DNA_DATA_New.to_csv("./DNA_DATA.csv")


# ================================================================================/

# RNA

RNA_DATA = pd.read_csv('../Data/Original/brca_rna_subtype.csv')

    # TODO: encode the data first
        # Vector of labels
# transposing
RNA_DATA_Trans = RNA_DATA.T
labelVector =  RNA_DATA_Trans.index.values
# removing "features"
labelVector= np.delete(labelVector,0)

# extracting the feature names:
RNA_features = list(RNA_DATA['Features'])
RNA_features.append('Class')


# Encoding Classes

def labelEncoding(arr):
    for i, e in enumerate(arr):
        if regexpA.search(e):
            arr[i]= '0'
        elif regexpB.search(e):
            arr[i]= '1'
        elif regexpT.search(e):
            arr[i]= '2'
        elif regexpH.search(e):
            arr[i]= '3'
        elif regexpU.search(e):
            arr[i]= '4'

    print(arr)
    return arr

labels = labelEncoding(labelVector)
        # featureMatrix
featureMatrix = RNA_DATA_Trans.iloc[1:,:].values

# Fusing the matrixes
RNA_DATA_New = np.insert(featureMatrix, 13195, labels, axis=1)

# Transforming into a dataFrame
RNA_DATA_New = pd.DataFrame(RNA_DATA_New)
# Adding the name of features:
RNA_DATA_New.columns = RNA_features
# RNA_DATA_New.to_csv("./RNA_DATA.csv")


# ================================================================================/

# CNV

CNV_DATA = pd.read_csv('../Data/Original/brca_cnv_subtype.csv')

    # TODO: encode the data first
        # Vector of labels
# transposing
CNV_DATA_Trans = CNV_DATA.T
labelVector =  CNV_DATA_Trans.index.values
# removing "features"
labelVector= np.delete(labelVector,0)

# extracting the feature names:
CNV_features = list(CNV_DATA['Features'])
CNV_features.append('Class')


# Encoding Classes
def labelEncoding(arr):
    for i, e in enumerate(arr):
        if regexpA.search(e):
            arr[i]= '0'
        elif regexpB.search(e):
            arr[i]= '1'
        elif regexpT.search(e):
            arr[i]= '2'
        elif regexpH.search(e):
            arr[i]= '3'
        elif regexpU.search(e):
            arr[i]= '4'

    print(arr)
    return arr

labels = labelEncoding(labelVector)
        # featureMatrix
featureMatrix = CNV_DATA_Trans.iloc[1:,:].values

# Fusing the matrixes
CNV_DATA_New = np.insert(featureMatrix, 15186, labels, axis=1)

# Transforming into a dataFrame
CNV_DATA_New = pd.DataFrame(CNV_DATA_New)
# Adding the name of features:
CNV_DATA_New.columns = CNV_features
# CNV_DATA_New.to_csv("./CNV_DATA.csv")


