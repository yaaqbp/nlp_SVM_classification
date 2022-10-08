"""
skrypt do wstepnego czyszczenia połączonego zbioru, na każdy wiersz mapowana jest funkcja clean_text, więc działa powoli. Szczegóły jak tekst jest czyszczony są w pliku preprocessing_utils.py
"""
import pandas as pd
import numpy as np

import sys
sys.path.append('../..')
import src.preprocessing_utils as pu
from src.utils import *
set_dir()
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
tqdm.pandas()
seed = seed_everything()
#import pickle

#df = pd.read_pickle('data/processed/arxiv_doaj_scimago_mapped_binarized.pkl')
df = pd.read_csv('data/processed/arxiv_doaj_scimago_mapped_binarized.csv', lineterminator='\n')
#df = df.iloc[:1000]
print('data loaded')
def clean_text(text):
    text = pu.basic_clean(text)
    text = pu.remove_whitespace(text)
    text = pu.accented_characters_removal(text)
    text = pu.lower_casing_text(text)
    text = pu.reducing_incorrect_character_repeatation(text)
    text = pu.expand_contractions(text)
    text = pu.removing_special_characters(text)
    text = pu.removing_numbers(text)
    text = pu.removing_stopwords(text)
    text = pu.lemmatization(text)
    return text

df.abstract = df.abstract.astype(str)
df.title = df.title.astype(str)
print('cleaning abstracts')
df['clean_abstract'] = df.abstract.progress_map(clean_text)
print('cleaning titles')sz
df['clean_title'] = df.title.progress_map(clean_text)

print('data cleaned')

df = df.drop(columns = ['abstract','title'], axis = 1)

df.drop_duplicates(subset = ['clean_abstract']).to_csv('data/processed/arxiv_doaj_scimago_mapped_binarized_cleaned.csv', index = False)
print('data saved in data/processed/arxiv_doaj_scimago_mapped_binarized_cleaned.csv')
#df.drop_duplicates(subset = ['clean_abstract']).to_pickle('data/processed/arxiv_doaj_scimago_mapped_binarized_cleaned.pkl')