"""
skrypt do wstępnego czyszczenia zbioru scimago. Sprawdza język abstraktów i tytułów, w razie potrzeby tłumaczy tytuł przy użyciu zewnętrznego api, więc jest dość wolny. 

"""
import pandas as pd
from itertools import count
import translators
import sys
sys.path.append('../..')
from src.utils import *
set_dir()
import warnings
warnings.filterwarnings('ignore')
seed = seed_everything()
import time
from tqdm import tqdm
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)


def detect_language(text):
    return nlp(text)._.language['language']

def detect_language_and_score(text):
    return nlp(text)._.language['language'], nlp(text)._.language['score']

def translate_to_en(text):
    language, score = detect_language_and_score(text)
    if language == 'en':
        print('en')
        return text
    if language != 'en' and score < .7:
        print('rather en')
        return text
    else:
        print('translating...')
        time.sleep(.1) # bez tego api odrzuca z racji zbyt dużej ilości zapytań na sekundę
        return translators.google(text, from_language = language, to_language = 'en')

    
    
df = pd.read_csv('data/raw/all_scimago.csv')
#df = df.iloc[:100]
df = df[['TITLE_OF_PUBLICATION','ABSTRACT','LANGUAGE','CAT_1','CAT_2','CAT_3','CAT_4']]
df = df.dropna(subset = ['ABSTRACT','CAT_1']).reset_index(drop = True)
df = df.drop_duplicates(subset = 'ABSTRACT')
print('detecting abstract language')
new_language = df.ABSTRACT.progress_map(detect_language)
df['new_language'] = new_language
df = df[df.new_language == 'en']
df.LANGUAGE = df.new_language
df = df.drop('new_language', axis = 1).reset_index(drop = True)
print('detecting and translatating title language')
title = df.TITLE_OF_PUBLICATION.progress_map(translate_to_en)
df['TITLE'] = title
df = df.drop('TITLE_OF_PUBLICATION', axis = 1)

df.to_csv('data/processed/scimago_cleaned.csv', index = False)
