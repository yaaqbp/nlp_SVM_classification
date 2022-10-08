"""
Map ArXiv dataset to common categories
"""

import json
import ast
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('../..')
from src.utils import *
set_dir()

def get_metadata():
    """
    Load ArXiv metadata
    """

    with open('data/external/arxiv/arxiv-metadata-oai-snapshot.json', 'r') as file:
        for line in file:
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
with open('data/dicts/arxiv_mapping_dict.txt') as file:
    data = file.read()
    mapping_dict = ast.literal_eval(data)

# List of target common categories
common_categories_array = pd.read_csv('data/dicts/labels_validated.csv', usecols=['eng']).values
common_categories = [x[0] for x in common_categories_array]


metadata = get_metadata()

with open('data/processed/arxiv_mapped.csv', 'w', encoding="utf-8") as file:
    # Write header of .csv file.  Format: "title;abstract;category_1;category_2;category_3..."
    file.write('title;abstract')
    for common_category in common_categories:
        file.write(f';{common_category}')
    file.write('\n')
    
    num_lines = sum(1 for line in open('data/external/arxiv/arxiv-metadata-oai-snapshot.json'))
    
    for i, paper in tqdm(enumerate(metadata), total=num_lines):
            paper_dict = json.loads(paper)

            title = clean_text(paper_dict['title'])
            abstract = clean_text(paper_dict['abstract'])

            file.write(f'{title};{abstract}')

            paper_categories = paper_dict['categories'].split()
            paper_categories_mapped = [mapping_dict[paper_category] for paper_category in paper_categories if paper_category in mapping_dict]
            # Iterate over target categories
            for common_category in common_categories:
                if common_category in paper_categories_mapped:
                    file.write(';1')
                else:
                    file.write(';0')
            file.write('\n')

print('Finished mapping ArXiv dataset.')