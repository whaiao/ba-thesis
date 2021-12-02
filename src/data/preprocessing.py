"""
Dataset preprocessing tool
Run from root.
"""

import re
from glob import glob
from functools import partial
from pprint import pprint
from typing import Tuple, List, Mapping

import pandas as pd
from reader import read_jsonlines

glob = partial(glob, recursive=True)


def rainbow_data(glob_path: str = './data/rainbow/**/train*') -> pd.DataFrame:
    datasets = glob(glob_path, recursive=True)
    corpus_tag_pattern = re.compile('\[\w+\]:')
    tag_open_pattern = re.compile('<\w+>')
    tag_closed_pattern = re.compile('</\w+>')
    
    def process_dataset(filepath: str) -> pd.DataFrame:
        d = pd.read_csv(filepath, encoding='utf8', index_col='index')

        def process_string(s: str) -> str:
            s = re.sub(corpus_tag_pattern, '', s)
            s = re.sub(tag_open_pattern, '', s)
            s = re.sub(tag_closed_pattern, '', s)
            s = re.sub('\n', '', s) 
            return s

        d['inputs'] = d['inputs'].apply(lambda x: process_string(x))
        return d[['inputs', 'targets']]

    df = process_dataset(datasets[0])
    for dataset in datasets[1:]:
        df.append(process_dataset(dataset))

    return df


def social_chem(glob_path: str = './data/processed/social_chem*') -> pd.DataFrame:
    datasets = glob(glob_path)
    cols = ['split', 'rot-categorization', 'rot-judgment', 'action', 'action-agree', 'situation', 'rot']
    
    df = pd.read_csv(datasets[0], sep='\t', encoding='utf8')[cols]
    for dataset in datasets[1:]:
        df.append(pd.read_csv(dataset, sep='\t', encoding='utf8')[cols])
        
    return df


def scruples(glob_path: str = './data/processed/*scruples*') -> Tuple[pd.DataFrame, pd.DataFrame]:
    datasets = glob(glob_path)
    
    def anecdotes(dataset: str) -> pd.DataFrame:
        cols = ['text', 'action', 'label', 'binarized_label']
        d = pd.read_csv(dataset, sep='\t', encoding='utf8')[cols]
        return d
        
    def dilemmas(dataset: str) -> pd.DataFrame:
        cols = ['actions', 'gold_label', 'controversial']
        d = pd.read_csv(dataset, sep='\t', encoding='utf8')[cols]

        def proc(e: List[Mapping[str, str]]) -> List[str]:
            return [i['description'] for i in eval(e)]

        d['actions'] = d['actions'].apply(lambda x: proc(x))
        return d
    
    df_anecdotes = [anecdotes(d) for d in datasets if 'anecdotes' in d]
    df_dilemmas = [dilemmas(d) for d in datasets if 'dilemmas' in d]
    df_anecdote = df_anecdotes[0]
    df_dilemma = df_dilemmas[0]

    for d in df_anecdotes[1:]:
        df_anecdote.append(d)

    for d in df_dilemmas[1:]:
        df_dilemma.append(d)

    return (df_anecdote, df_dilemma)

def moral_stories(glob_path: str = './data/moral_stories_datasets/**/*.jsonl') -> Tuple[pd.DataFrame, pd.DataFrame]:
    datasets = glob(glob_path)
    d = read_jsonlines(datasets[0])
    full_df = pd.DataFrame(d)
    task, categories = datasets[1].split('/')[3:5]
    mapped_df = pd.DataFrame(read_jsonlines(datasets[1]), index=None)
    mapped_df.insert(0, 'Task', task)
    mapped_df.insert(1, 'Category', categories)

    for dataset in datasets[2:]:
        task, categories = dataset.split('/')[3:5]
        tmp = pd.DataFrame(read_jsonlines(dataset), index=None)
        tmp['Task'] = task
        tmp['Category'] = categories
        mapped_df.append(tmp)

    return (full_df, mapped_df)


