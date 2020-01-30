#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random


# In[3]:


def my_data_loader(file_name, folder='', train_label=None, event_time=10, nrows=60):
    real_list = [30,1154,1168]
    file_id = int(file_name.split('.')[0]) # file id만 불러오기
    df = pd.read_csv(folder+file_name, index_col=0, nrows=nrows) # 파일 읽어오기
    # Bad, CLOSE, Equip Fail, No Data, Normal, OFF, ON, OPEN, System.Char[] 등 문자열 데이터를 포함
    df = df.replace('.*', 0.1, regex=True).fillna(0.1) # 모든 문자열과 NA값을 0으로 대체
    random_time = random.randrange(1,16)
    if file_id in real_list:
        df = df.loc[event_time:]
    else:
        df = df.loc[random_time:] # event_time 이후의 row들만 가지고 오기
    df.index = np.repeat(file_id, len(df)) # row 인덱스를 file id로 덮어 씌우기 
    if type(train_label) != type(None):
        label = train_label.loc[file_id]['label'] 
        df['label'] = np.repeat(label, len(df)) #train set일 경우 라벨 추가하기 
    return df

# In[ ]:




