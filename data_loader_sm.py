#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np

def eu_distance(data):
        len_list = []
        length = 15
        for i in range(length-1):
            dis = (((data.loc[i] - data.loc[i+1])**2)**0.5).sum()
            len_list.append(dis)
        event = (len_list.index(max(len_list)))+1
        return event
    
def replace_out(df):
        for i in df.columns:
            q1, q3 = np.percentile(df[i],[25,75])
            IQR = q3 - q1
            lower = q1 - ( IQR * 1.5 )
            upper = q3 + ( IQR * 1.5 )
            fill_val = df[i].median()
            df[i][ (df[i]>upper) | (df[i]<lower) ] = fill_val
            fill_val = 0 
        return df
    
def data_loader_sm(file_name, folder='', train_label=None, event_time=0, nrows=60):
    file_id = int(file_name.split('.')[0]) # file id만 불러오기
    df = pd.read_csv(folder+file_name, index_col=0, nrows=nrows) # 파일 읽어오기
    df = df.replace('.*', 0, regex=True).fillna(0) # 모든 문자열과 NA값을 0으로 대체, 또한 문자열이 포함된 데이터는 전체 데이터 중 train = 1, test = 2개밖에 없음
    event_time = eu_distance(df)
    df = df.loc[event_time:] #event_time 이후의 row들만 가지고 오기
    df = replace_out(df)
    df.index = np.repeat(file_id, len(df)) # row 인덱스를 file id로 덮어 씌우기 
    if type(train_label) != type(None):
        label = train_label.loc[file_id]['label'] 
        df['label'] = np.repeat(label, len(df)) #train set일 경우 라벨 추가하기
    return df

