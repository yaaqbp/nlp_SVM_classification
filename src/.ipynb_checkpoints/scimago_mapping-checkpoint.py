#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MultiLabelBinarizer


# In[2]:


scimago = pd.read_csv('../data/processed/scimago_cleaned.csv')


# In[57]:


labels = pd.read_csv('mapowanie.csv')


# In[58]:


def create_map_dict(labels):
    labels = labels[['eng','old_labels','old_labels_1','old_labels_2','old_labels_3','old_labels_4']].replace('0', float('nan')).dropna(subset = ['old_labels']).fillna('0').reset_index(drop = 1)
    for col in labels.columns:
        labels[col] = labels[col].str.strip().str.lower()
    dct = dict()
    for row in labels.iterrows():
        row = row[1]
        for i in range(1,len(row)):
            if row[i] != '0':
                dct[row[i]] = row[0]
    return dct


# In[59]:


def map_or_delete(df, subset, dct):
    uniques = []
    for col in subset:
        uniques+=list(df[col].unique())
    uniques = set(uniques)
    empty_labels = list(uniques - set(dct.keys()))
    for label in empty_labels:
        dct[label] = np.nan
    df = df.replace(dct)
    df = df.dropna(subset = subset, how = 'all')
    return df


# In[60]:


dct = create_map_dict(labels)


# In[61]:


CATS = ['CAT_1', 'CAT_2', 'CAT_3', 'CAT_4']


# In[62]:


for col in CATS:
    scimago[col] = scimago[col].str.strip().str.lower()


# In[63]:


scimago = map_or_delete(scimago, CATS, dct)


# In[64]:


scimago['CAT'] = scimago[CATS].stack().groupby(level = 0).agg(list)
scimago = scimago.dropna(subset = ['CAT']).reset_index(drop = True)


# In[65]:


scimago['CATstr'] = scimago.CAT.map(lambda x: ' '.join(x))
vc = scimago.CATstr.value_counts()
vc = vc[vc>2]
scimago = scimago[scimago.CATstr.isin(vc.index.tolist())]
scimago = scimago.drop('CATstr', axis = 1).reset_index(drop = True)


# In[66]:


mlb = MultiLabelBinarizer()
CAT = mlb.fit_transform(scimago['CAT'])
scimago = scimago.join(pd.DataFrame(CAT, columns = list(mlb.classes_)))


# In[67]:


scimago.drop(CATS, axis = 1).to_csv('../data/processed/scimago_mapped_binarized.csv', index = False)

