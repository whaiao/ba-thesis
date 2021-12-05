#! /usr/bin/env python3
"""
Configuration handling for experiments
"""
from ast import literal_eval
from functools import reduce
from pathlib import Path
from typing import List, Union, Mapping, Callable, Iterable, TypeVar

import jax.numpy as jnp
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from multiprocess.pool import Pool, AsyncResult

import yaml

Dataframe = pd.DataFrame
T = TypeVar('T', np.ndarray, jnp.ndarray, torch.Tensor)

# def get_config(yaml_filepath: str) -> Mapping[str, Union[str, float, int]]:
#     """
#     Parses YAML file and returns config dictionary
#
#     Args:
#         yaml_filepath: file path to yaml config
#
#     Returns:
#         A dictionary of experiment configuration settings
#     """
#     with open(yaml_filepath, 'rb') as f:
#         return yaml.safe_load(f)


def return_tensor(x: np.ndarray, return_as: str, dtype=None) -> T:
    if return_as == 'torch':
        return torch.tensor(x, dtype=dtype)
    elif return_as == 'numpy':
        return np.array(x, dtype=dtype)
    elif return_as == 'jax':
        return jnp.array(x, dtype)
    else:
        raise ValueError(f'{dtype} is not supported.')


def multiprocess(f: Callable, args: Iterable, n_workers=None) -> AsyncResult:
    with Pool(n_workers) as p:
        res = p.starmap_async(f, args)
        return res.get()


def remap_dataframe_dtypes(df: Dataframe, cols: List[str]) -> Dataframe:
    for c in tqdm(cols):
        tmp_col = df[c].apply(lambda x: literal_eval(x))
        for d in tmp_col:
            if len(d) > 1:
                new_col = reduce(lambda x, y: x + literal_eval(y), d, [])
                d = new_col
            else:
                continue
        df[c] = tmp_col

    # IMPORTANT: otherwise dataframe will be saved with strings again
    df.to_pickle('data/tmp/converted.pickle')

    # df.to_csv('data/tmp/converted.tsv', sep='\t', encoding='utf8')
    return df
