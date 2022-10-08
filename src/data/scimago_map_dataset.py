import sys
sys.path.append('../..')
from src.utils import *
set_dir()

import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MultiLabelBinarizer
filepath = 'data/processed/scimago_cleaned.csv'
labelspath = 'data/dicts/scimago_mapping_dict.csv'

scimago = pd.read_csv(filepath)
labels = pd.read_csv(labelspath)

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

dct = create_map_dict(labels)


CATS = ['CAT_1', 'CAT_2', 'CAT_3', 'CAT_4']

for col in CATS:
    scimago[col] = scimago[col].str.strip().str.lower()
    
scimago = map_or_delete(scimago, CATS, dct)
print('scimago_mapped')

scimago['CAT'] = scimago[CATS].stack().groupby(level = 0).agg(list)
scimago = scimago.dropna(subset = ['CAT']).reset_index(drop = True)

mlb = MultiLabelBinarizer()
CAT = mlb.fit_transform(scimago['CAT'])
scimago = scimago.join(pd.DataFrame(CAT, columns = list(mlb.classes_)))

scimago = scimago.drop(CATS+['LANGUAGE', 'CAT'], axis = 1)
scimago = scimago.rename(columns = {'ABSTRACT':'abstract','TITLE':'title'})
scimago.to_csv('data/processed/scimago_mapped_binarized.csv', index = False)

print('scimago_mapped_binarized.csv saved succesfully')


