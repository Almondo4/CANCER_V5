
import pandas as pd
import numpy as np


# Hybrid
Hybrid_Data = pd.read_csv('../Hybrid_DATA.csv')
HYBRIDCSV = pd.read_csv('../HYBRID.csv')

featureMatrix = Hybrid_Data.iloc[:,1:-1]
labelVector = Hybrid_Data.iloc[:,-1]
features = Hybrid_Data.columns


# Rmoving the Unnamed
try:
    Hybrid_Data.drop('Unnamed: 0', inplace=True, axis=1)
except :
    print('already Dealt with')

# Creating Baseline
Hybrid_Data = Hybrid_Data.sample(frac=1,random_state=200)

Hybrid_Data.to_csv('./B_HYBRID.csv',index=False)

# DNA
DNA_DATA = pd.read_csv("../DNA_DATA.csv")
featureMatrixDNA = DNA_DATA.iloc[:,:-1]
labelVectorDNA = DNA_DATA.iloc[:,-1]
feauresDNA = DNA_DATA.columns

# Creating Baseline DNA
DNA_DATA = DNA_DATA.sample(frac=1,random_state=200)
DNA_DATA.to_csv('./B_DNA.csv',index=False)

# CNV


CNV_DATA = pd.read_csv('../CNV_DATA.csv')
featureMatrixCNV = CNV_DATA.iloc[:,:-1]
labelVectorCNV = CNV_DATA.iloc[:,-1]
featuresCNV = CNV_DATA.columns


# Creating Baseline CNV

CNV_DATA = CNV_DATA.sample(frac=1,random_state=200)
CNV_DATA.to_csv('./B_CNV.csv',index=False)


# RNA

RNA_DATA = pd.read_csv('../CNV_DATA.csv')
featureMatrixRNA = RNA_DATA.iloc[:,:-1]
labelVectorRNA = RNA_DATA.iloc[:,-1]
featuresRNA = RNA_DATA.columns

# Creating Baseline RNA
RNA_DATA = RNA_DATA.sample(frac=1,random_state=200)
RNA_DATA.to_csv('./B_RNA.csv',index=False)

