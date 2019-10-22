# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:01:40 2019

@author: Gayathri
"""

'''
Creates submission csv from prediction csv

'''
import os
import pandas as pd
import re
pred_df = pd.read_csv('results_Adam.csv', low_memory = False)
submission = pd.read_csv('sample_submission (1).csv',low_memory = False)
submission = submission.iloc[0:0]
pred_df['inkml_file_nbr']= pred_df['Filename'].apply(lambda x:str(re.split('_',x,3)[2]))
pred_df['inkml_file_name']= 'hack_data/test/test_'+pred_df['inkml_file_nbr']+'.inkml'
files = list(set(pred_df['inkml_file_name'].tolist()))
s = submission.columns.tolist()
#submission = submission.iloc[0:0]
for idx,i in enumerate(files):
    new_df = pred_df[pred_df['inkml_file_name']==i]
    d = dict.fromkeys(submission.columns.tolist(), 0)
    for index,row in new_df.iterrows():
        if row['Predictions'] in s:
            curr_count = d.get(row['Predictions'])
            curr_count = curr_count+1
            d[row['Predictions']] = curr_count
        elif '\\'+str(row['Predictions']) in s:
            curr_count = d.get('\\'+str(row['Predictions']))
            curr_count = curr_count+1
            d['\\'+str(row['Predictions'])] = curr_count
        else:
            print(row['Predictions'],'NOT FOUND')
            print('ERROR')
            break
    d['file_path']=i
    df1  = pd.DataFrame(d,index=[0])
    submission = pd.concat([submission, df1])
    print(idx)
slash_columns = [r'\alpha',r'\beta',r'\cos',r'\div',r'\exists',r'\forall','\gamma','\geq','\gt','\in','\infty','\int','\ldots','\leq','\lim','\log','\lt',r'\neq','\phi','\pi','\pm',r'\rightarrow','\sin','\sqrt','\sum',r'\tan',r'\theta',r'\times','\{','\}']
no_slash_column = []
for  i in slash_columns:
    no_slash_column.append(re.sub(r"\\",'',i)) 
dictionary = dict(zip(no_slash_column, slash_columns))
submission.drop([slash_columns],axis = 1)
submission.rename(columns=dictionary, inplace=True)
submission.to_csv('submission_mfdm_ai_adam.csv', index = False)
       
        