"""
Map DOAJ dataset to common categories
"""

import sys
sys.path.append('../..')
from src.utils import *
set_dir()


import os
import json
import ast
import pandas as pd
from tqdm import tqdm


def get_metadata(filepath):
    """
    Load DOAJ metadata from file
    """
    with open(filepath, 'r') as f:
        for line in f:
            yield line

def clean_text(text):
    """
    Clean string from '\n', '"' and ';' characters
    """
    cleaned_text = text.replace('\n', ' ')
    cleaned_text = cleaned_text.replace('\"', '')
    cleaned_text = cleaned_text.replace('"', '')
    cleaned_text = cleaned_text.replace(';', '')
    
    return cleaned_text



# Dictionary used to map ArXiv categories to common categories
with open('data/dicts/doaj_mapping_dict.txt') as file:
    data = file.read()
    mapping_dict = ast.literal_eval(data)

# List of target common categories
common_categories_array = pd.read_csv('data/dicts/labels_validated.csv', usecols=['eng']).values
common_categories = [x[0] for x in common_categories_array]

directory = 'data/external/doaj/'

with open('data/processed/doaj_mapped.csv', 'w', encoding="utf-8") as file:
    # Write header of .csv file.  Format: "title;abstract;category_1;category_2;category_3..."
    file.write('title;abstract')
    for common_category in common_categories:
        file.write(f';{common_category}')
    file.write('\n')

    for _, filename in tqdm(enumerate(os.listdir(directory)), total=len(os.listdir(directory))):
        filepath = os.path.join(directory, filename)

        metadata = get_metadata(filepath)

        num_lines = sum(1 for line in open(filepath, encoding='utf-8'))

        for i, row in enumerate(metadata):
            # .json files begin and end with brackets so its necessary to skip them with string slicing when parsing
            if i == 0:
                json_ = json.loads(row[1:-2])
            elif i == num_lines - 1:
                json_ = json.loads(row[0:-1])
            else:
                json_ = json.loads(row[0:-2])

            # Only process papers in english
            if 'EN' in json_['bibjson']['journal']['language']:
                # Skip papers with missing abstract, title or categories
                if 'abstract' in json_['bibjson'] and 'title' in json_['bibjson'] and 'subject' in json_['bibjson']:
                    title = clean_text(json_['bibjson']['title'])
                    abstract = clean_text(json_['bibjson']['abstract'])
                    
                    # Skip rows with no categories
                    # if any([(x in mapping_dict.keys()) for x in paper_categories]):

                    file.write(f'{title};{abstract}')

                    paper_categories = [subject['term'] for subject in json_['bibjson']['subject']]
                    paper_categories_mapped = [mapping_dict[cat] if cat in mapping_dict else 'PLACEHOLDER' for cat in paper_categories]

                    for common_category in common_categories:
                        if common_category in paper_categories_mapped:
                            file.write(';1')
                        else:
                            file.write(';0')
                    file.write('\n')

print('Finished mapping DOAJ dataset.')