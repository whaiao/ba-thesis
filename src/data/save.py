#! /usr/bin/env python3
"""
All functions to process data go here
"""
from glob import glob
from pickle import dump

import pandas as pd
from tqdm import tqdm

from src_old.data.reader import read_jsonlines, read_csv


def jsonl2tsv(datapath: str = 'data/', saveto: str = 'data/processed/'):
    files = glob(f'{datapath}**/*.jsonl', recursive=True)
    for file in tqdm(files):
        filename = file.split('/')[-1].split('.')
        filename = filename[:2] if len(filename) > 2 else filename[0]
        new_filename = saveto + '_'.join(filename) + '.tsv'
        data = read_jsonlines(file)
        pd.DataFrame(data).to_csv(new_filename, encoding='utf8', sep='\t')


def csv2tsv(datapath: str = 'data/', saveto: str = 'data/processed/'):
    files = glob(f'{datapath}**/*.csv')
    for file in tqdm(files):
        filename = file.split('/')[-1].split('.')[0]
        new_filename = saveto + filename + '.tsv'
        data = read_csv(file)
        pd.DataFrame(data).to_csv(new_filename, encoding='utf8', sep='\t')


def to_pickle(data, saveto: str = 'data/processed/'):
    with open(saveto, 'wb') as p:
        dump(data, p)
    print(f'Saved pickle file to: {saveto}')
