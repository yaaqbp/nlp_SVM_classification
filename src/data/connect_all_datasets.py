import pandas as pd

import sys
sys.path.append('../..')
from src.utils import *
set_dir()

arxiv = pd.read_csv('data/processed/arxiv_mapped.csv', sep = ';',encoding="utf-8",lineterminator='\n')
scimago = pd.read_csv('data/processed/scimago_mapped_binarized.csv',sep = ',',encoding="utf-8",lineterminator='\n')
doaj = pd.read_csv('data/processed/doaj_mapped.csv', sep = ';',encoding="utf-8",lineterminator='\n')

labels = pd.read_csv('data/dicts/labels_validated.csv')
labels = labels.eng.dropna().str.lower().str.strip()

def check_labels(df, labels):
    df.columns = df.columns.str.lower().str.strip()
    df_labels = df.drop(columns = ['abstract','title']).columns
    df[list(set(labels)-set(df_labels))] = 0
    df = df.drop(columns = list(set(df_labels)-set(labels)))
    if df.shape[1] == 218:
        return df
    else:
        raise ValueError
        
scimago = check_labels(scimago, labels)
arxiv = check_labels(arxiv, labels)
doaj = check_labels(doaj, labels)

print('scimago empty rows:',(scimago.drop(columns = ['title','abstract']).sum(axis = 1)==0).sum())
print('arxiv empty rows:',(arxiv.drop(columns = ['title','abstract']).sum(axis = 1)==0).sum())
print('doaj empty rows:',(doaj.drop(columns = ['title','abstract']).sum(axis = 1)==0).sum())

print('droping empty rows...')
scimago = scimago[scimago.drop(columns = ['title','abstract']).sum(axis = 1)>0].reset_index(drop = 1)
arxiv = arxiv[arxiv.drop(columns = ['title','abstract']).sum(axis = 1)>0].reset_index(drop = 1)
doaj = doaj[doaj.drop(columns = ['title','abstract']).sum(axis = 1)>0].reset_index(drop = 1)

print('scimago empty rows:',(scimago.drop(columns = ['title','abstract']).sum(axis = 1)==0).sum())
print('arxiv empty rows:',(arxiv.drop(columns = ['title','abstract']).sum(axis = 1)==0).sum())
print('doaj empty rows:',(doaj.drop(columns = ['title','abstract']).sum(axis = 1)==0).sum())

df = pd.concat([arxiv, scimago, doaj], axis = 0)

#print(arxiv.shape)
#print(scimago.shape)
#print(doaj.shape)
if scimago.shape[0]+arxiv.shape[0]+doaj.shape[0] == len(df):
    print('shape correct, datasets connected')

df = df.dropna(subset = ['abstract']).drop_duplicates(subset = 'abstract').reset_index(drop = True)

df.to_csv('data/processed/arxiv_doaj_scimago_mapped_binarized.csv',index = False, encoding="utf-8")

print('connected datasets saved in data/processed/arxiv_doaj_scimago_mapped_binarized.csv')