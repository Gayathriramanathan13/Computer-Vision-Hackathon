# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:03:49 2019

@author: Gayathri
"""

import os
import sys
import pandas as pd
basedir = r'gayathri-ramanathan-predict-mathnotations-master/hack_data/extracted_data_train_test/'
os.chdir(os.getcwd()+basedir)
path = 'train_imgs2/'
def get_immediate_subdirectories(a_dir):
    return[name for name in os.listdir(a_dir) 
            if os.path.isdir(os.path.join(a_dir,name))]
    
lis = get_immediate_subdirectories(str(path))
dic = {}
N=0
for i in lis:
    l = len(os.listdir(str(path)+str(i)))
    dic[i] = l
    N= N+l
print('Total Images for Training:',N)
df_counts = pd.DataFrame(dic.items(),columns = ['Class','Count'])
df_counts.to_csv('count_per_class.csv')