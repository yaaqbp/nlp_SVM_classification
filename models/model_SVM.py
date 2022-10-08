
import pandas as pd
import numpy as np

import sys
sys.path.append('..')
import src.preprocessing_utils as pu
#import metrics_utils as mu

from src.utils import *
import warnings
warnings.filterwarnings('ignore')

seed = seed_everything()
set_dir()


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime

import joblib


# In[3]:


df = pd.read_csv('data/processed/arxiv_doaj_scimago_mapped_binarized_cleaned.csv')
# można spróbować na całości jeśli ram pozwoli
N = 2*10**6
print(f'{N} sample of {df.shape[0]} dataframe')
df = df.sample(n = N).reset_index(drop = True)



df['text'] = df['clean_title'] + ' ' + df['clean_abstract']
df = df.dropna(subset = ['text'])
labels = df.drop(columns = ['text','clean_title','clean_abstract']).columns
df[labels] = df[labels].astype(np.int8)
y_new = pd.DataFrame()
for col in labels:
    y_new[col] = df[col].replace({0:np.nan,1:col})
df['CAT'] = y_new.stack().groupby(level = 0).agg(list)
df['CATstr'] = df.CAT.map(lambda x: ' '.join(x))
vc = df.CATstr.value_counts()
vc = vc[vc>1]
df = df[df.CATstr.isin(vc.index.tolist())]
df = df.drop('CATstr', axis = 1).reset_index(drop = True)
y = df.drop(['text','clean_title','clean_abstract','CAT'],axis = 1)
X = df['text']
empty_labels = list(y.sum()[y.sum() == 0].index)
y = y.drop(empty_labels, axis = 1)
CAT = y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .33, random_state = seed, stratify = CAT)

def score(y_true, y_pred, index):
    """Calculate precision, recall, and f1 score"""
    

    metrics = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    performance = {'precision': metrics[0], 'recall': metrics[1], 'f1': metrics[2]}
    return pd.DataFrame(performance, index=[index])

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range = (1, 2))),
    ('clf',  MultiOutputClassifier(estimator=LinearSVC(class_weight='balanced',random_state=seed, verbose = True, C = 1, max_iter = 100000), n_jobs = 1))])

print(X.shape, y.shape)
scores = pd.DataFrame()
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
machine_learning = score(y_test, y_pred, 'semi_optimized_SVM')
scores = pd.concat([scores, machine_learning])

x = (str(datetime.now())).replace(' ','_')[:-7]
joblib.dump(pipeline, f'models/pipeline_{x}.sav')
joblib.dump(y.columns, f'models/labels_{x}.sav')
scores.to_csv(f'reports/experiment_{x}.csv')

