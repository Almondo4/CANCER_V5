

import pandas as pd
import numpy as np


L1_hybrid = pd.read_csv('../../Data/L1_Enhancement/S1_Hybrid_os_T1.csv')


index_names = L1_hybrid[ L1_hybrid['class'] == 4 ].index

L1_currated= L1_hybrid.drop(index_names)

L1_currated.to_csv('L1_Currated.csv',index=False)


#  KNN Plotting

