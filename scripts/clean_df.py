from copy import deepcopy
from typing import Dict, List, Union

import pandas as pd
from tqdm import tqdm

from src.utils import multiprocess

Dataframe = pd.DataFrame


def clean_df(df: Union[Dataframe, str], columns: List[str], multiprocessing: bool = False) -> Dataframe:
    if isinstance(df, str):
        df = pd.read_pickle(df)

    if multiprocessing:
        args = [[df[c], c] for c in columns] 
        res = multiprocess(process_col, args)
        print(res)
        for d in res:
            for k, v in d:
                df[k] = v

    for t, col in enumerate(columns):
        print(f'Processing col: {col}\t{t} of {len(columns)}')
        for l in tqdm(df[col]):
            for i in l:
                i = pd.eval(i)

    df.to_pickle('data/tmp/update.pickle')
    return df


#     for col in columns:
#         print(f'Working on {col}')
#         current_col = df[col]
#         for l in tqdm(current_col):
#             for i in l:
#                 i = pd.eval(i)
#         df[col] = current_col
#         print(type(df[col][0][0]))

def process_col(series: pd.Series, col: str) -> Dict[str, pd.Series]:
    for l in series:
        for i in l:
            i = pd.eval(i)
    
    return {col: series}


if __name__ == "__main__":
    clean_df('data/tmp/converted.pickle', 
            columns=[
                'oEffect',
                'oReact',
                'oWant',
                'xAttr',
                'xEffect',
                'xIntent',
                'xNeed',
                'xReact',
                'xWant',
                'prefix',
                ])
