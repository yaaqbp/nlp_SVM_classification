#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
import spacy
from spacy_langdetect import LanguageDetector
from spacy.language import Language
import translators
import sys
sys.path.append('..')
import preprocessing_utils as pu
from src.utils import *
import warnings
warnings.filterwarnings('ignore')

seed = seed_everything()

def get_lang_detector(nlp, name):
    return LanguageDetector() 


nlp = spacy.load('en_core_web_sm') 
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)


def check_language(text):
    doc = nlp(text)    
    detect_language = doc._.language
    return detect_language['language']


# In[69]:


def translate_to_en(text):
    language = nlp(text)._.language['language']
    if  language == 'en':
        return text
    else:
        return translators.google(text, from_language = language, to_language = 'pl')


# In[89]:


df = pd.read_csv('../data/raw/all_scimago.csv')
df = df.iloc[:1000]


# In[71]:


df = df[['AUTHOR_NATURAL_ID','Title','ABSTRACT','LANGUAGE','CAT_1','CAT_2','CAT_3','CAT_4']]


# In[72]:


df = df.dropna(subset = ['ABSTRACT','CAT_1']).reset_index(drop = True)


# In[73]:


df = df.drop_duplicates(subset = 'ABSTRACT')


# In[74]:




# In[75]:


new_language = df.ABSTRACT.map(check_language)


# In[76]:


df['spacy_language'] = new_language
df = df[df.spacy_language == 'en']
df.LANGUAGE = df.spacy_language
df = df.drop('spacy_language', axis = 1).reset_index(drop = True)


# In[84]:


title = df.Title.map(translate_to_en)
df['TITLE'] = title
df = df.drop('Title', axis = 1)


# In[88]:


df.to_csv('../data/processed/scimago_cleaned.csv', index = False)

