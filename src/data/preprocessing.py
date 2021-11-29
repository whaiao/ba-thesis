import re
from glob import glob

import pandas as pd



SEPTOKEN = ' [SEP] '
CLSTOKEN = '[CLS] '


def rainbow_data(glob_path: str = './data/rainbow/**/train*'):
    datasets = glob(glob_path, recursive=True)
    corpus_tag_pattern = re.compile('\[\w+\]:')
    tag_open_pattern = re.compile('<\w+>')
    tag_closed_pattern = re.compile('</\w+>')
    
    def process_dataset(filepath: str) -> pd.DataFrame:
        d = pd.read_csv(filepath, encoding='utf8', index_col='index')

        def process_string(s: str) -> str:
            s = re.sub(corpus_tag_pattern, CLSTOKEN, s)
            s = re.sub(tag_open_pattern, '', s)
            s = re.sub(tag_closed_pattern, SEPTOKEN, s)
            s = re.sub('\n', '', s) 
            return s

        d['inputs'] = d['inputs'].apply(lambda x: process_string(x))
        return d[['inputs', 'targets']]

    for dataset in datasets:
        d = process_dataset(dataset)

rainbow_data()
