# ?here i will combine

import pandas as pd
import numpy as np

# hybrid_s1= pd.read_csv('S1_Hybrid_os_T1.csv')

RNA_DATA = pd.read_csv('../S_RNA_os.csv')
DNA_DATA = pd.read_csv('../S_DNA_os.csv')
CNV_DATA = pd.read_csv('../S_CNV_os.csv')





## Sorting accordingto class

RNA_DATA_Sorted = RNA_DATA.sort_values('class')
DNA_DATA_Sorted = DNA_DATA.sort_values('class')
CNV_DATA_Sorted = CNV_DATA.sort_values('class')


# DNA FEATURES
DNA_Features = list(DNA_DATA.columns)
DNA_Features.remove('class')

# RNA FEATURES
RNA_Features = list(RNA_DATA.columns)
RNA_Features.remove('class')

#CNV Features
CNV_Features = list(CNV_DATA.columns)
CNV_Features.remove('class')



# Extrating Feature MAtrix

# RNA_DATA_2c = RNA_DATA_Sorted.drop('class',  axis=1)
# DNA_DATA_2c = DNA_DATA_Sorted.drop('class',  axis=1)
# CNV_DATA_2c = CNV_DATA_Sorted.drop('class',  axis=1)
#



        # featureMatrix
featureMatrixDNA = RNA_DATA_Sorted.iloc[:,:-1].values
featureMatrixRNA = DNA_DATA_Sorted.iloc[:,:-1].values
featureMatrixCNV = CNV_DATA_Sorted.iloc[:,:-1].values

        # Vector of labels
labelVector =  RNA_DATA_Sorted.iloc[:,-1].values

# Creating the Hybrid Feature Matrix
Hybrid = np.hstack((featureMatrixDNA,featureMatrixRNA,featureMatrixCNV))
HybridFrame = np.insert(Hybrid,3157,labelVector,axis=1)
HybridFrame = pd.DataFrame(HybridFrame)



# naming Appropriately:

    # DNA
for i in range(DNA_Features.__len__()):
    DNA_Features[i]=DNA_Features[i]+"_DNA"

    # RNA
for i in range(RNA_Features.__len__()):
    RNA_Features[i]=RNA_Features[i]+"_RNA"

    # CNV
for i in range(CNV_Features.__len__()):
    CNV_Features[i]=CNV_Features[i]+"_CNV"

# Column creation

mergedLabels = np.hstack((DNA_Features,RNA_Features,CNV_Features,['class']))
HybridFrame.columns = mergedLabels

HybridFrame.to_csv("./S1_Hybrid_T2.csv",index=False) # <-----------------------------------

# S1_Hybrid_T2 _ Training
#
# s1_hybrid= pd.read_csv('../../Baseline/B_HYBRID.csv')
# mergedLabels = np.hstack((DNA_Features,RNA_Features,CNV_Features,['Class']))
#
#
#
#
# # Combining the datasets
#
#
#
#
# s1_hybrid = s1_hybrid[mergedLabels]
#
# # # S1_Hybrid_T2 _ Testing
# # s1_hybrid_test= pd.read_csv('../../Baseline/Testing/test.csv')
# # s1_hybrid_test= s1_hybrid_test [mergedLabels]
#
# #Saving The Datasets
# s1_hybrid.to_csv('S1_hybrid_T2.csv',index=False)
# # s1_hybrid_test.to_csv('S1_hybrid_test_T2.csv',index=False)
#
#
# # Another way of combining