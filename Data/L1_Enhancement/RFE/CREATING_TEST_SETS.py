
import pandas as pd


# S1_Hybrid_T1 _ Testing

## Extracting Features
s1_hybrid= pd.read_csv('../Training/S1_Hybrid_os_T1.csv')
s1_hybrid_Features= list(s1_hybrid.columns)

## Subset

s1_hybrid_test= pd.read_csv('../../Baseline/Testing/test.csv')

# Because s1 hybrid class vector is Class instead of class I will do the following:

s1_hybrid_Features.remove('class')

Class_labels= s1_hybrid_test.pop('Class')

s1_hybrid_test=s1_hybrid_test[s1_hybrid_Features]

s1_hybrid_test= s1_hybrid_test.assign(Class=Class_labels)



s1_hybrid_test.to_csv('S1_Hybrid_test_T1.csv')


# S1_Hybrid_T2 _ Testing

## ----> created together with the training set


# #----------------------------------------DNA

# S_DNA_Testing

## Extracting Features
s1_DNA= pd.read_csv('../Training/S_DNA_os.csv')
s1_DNA_Features= list(s1_DNA.columns)
s1_DNA_Features.remove('class') # Again Because of Uppercase C


## Subset

s1_DNA_test= pd.read_csv('../../Baseline/Testing/DNA_test.csv')
s1_DNA_test=s1_DNA_test[s1_DNA_Features]
s1_DNA_test= s1_DNA_test.assign(Class=Class_labels) #  Uppercase C

s1_DNA_test.to_csv('S_DNA_test.csv')


# #----------------------------------------RNA

# S_RNA_Testing

## Extracting Features
s1_RNA= pd.read_csv('../Training/S_RNA_os.csv')
s1_RNA_Features= list(s1_RNA.columns)
s1_RNA_Features.remove('class')#  Uppercase C

## Subset

s1_RNA_test= pd.read_csv('../../Baseline/Testing/RNA_test.csv')
s1_RNA_test=s1_RNA_test[s1_RNA_Features]
s1_RNA_test= s1_RNA_test.assign(Class=Class_labels) #  Uppercase C

s1_RNA_test.to_csv('S_RNA_test.csv')



# #----------------------------------------CNV

# S_CNV_Testing

## Extracting Features
s1_CNV= pd.read_csv('../Training/S_CNV_os.csv')
s1_CNV_Features= list(s1_CNV.columns)
s1_CNV_Features.remove('class')#  Uppercase C

## Subset

s1_CNV_test= pd.read_csv('../../Baseline/Testing/CNV_test.csv')
s1_CNV_test=s1_CNV_test[s1_CNV_Features]
s1_CNV_test= s1_CNV_test.assign(Class=Class_labels) #  Uppercase C

s1_CNV_test.to_csv('S_CNV_test.csv')