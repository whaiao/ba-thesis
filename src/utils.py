#! /usr/bin/env python3
"""
General utils for project
"""
from functools import partial
from multiprocessing import cpu_count
from typing import List, Union, Callable, Iterable, TypeVar

import jax.numpy as jnp
from multiprocess.pool import Pool, AsyncResult
import numpy as np
import pandas as pd
import torch
from torch import Tensor

Dataframe = pd.DataFrame

T = TypeVar('T', np.ndarray, jnp.ndarray, torch.Tensor)


def return_tensor(x: np.ndarray, return_as: str, dtype=None):
    if return_as == 'torch':
        return torch.tensor(x, dtype=dtype)
    elif return_as == 'numpy':
        return np.array(x, dtype=dtype)
    elif return_as == 'jax':
        return jnp.array(x, dtype)
    else:
        raise ValueError(f'{dtype} is not supported.')


read_tsv = partial(pd.read_csv, sep='\t', encoding='utf8')
sorted_dict = partial(sorted, key=lambda item: item[1], reverse=True)


def multiprocess_multiargs(
    f: Callable, args: Iterable, n_workers=cpu_count()) -> AsyncResult:
    with Pool(n_workers) as p:
        res = p.starmap_async(f, args)
        return res.get()


def multiprocess_dataset(f: Callable, dataset: Dataframe,
                         **kwargs) -> List[Dataframe]:
    """Multiprocess datasets with number of available CPUs.

    Args:
        f - function to apply to datasets
        dataset - dataset to split in `size // number of cpus`
        kwargs - kwargs to fixate function call

    Returns:
        multiprocessing result
    """

    # fixate function arguments to process with multiprocessing, if fixating args
    # not suitable check `multiprocess_multiargs`
    fn = partial(f, **kwargs)

    def split_jobs(dataset: Dataframe,
                   n_workers: int = 4) -> Union[List[Dataframe], int, int]:
        """Split dataset to size of number of cpus available"""
        df = dataset
        n_cpus = cpu_count() if n_workers is None else n_workers
        sets = []
        split_at = len(df) // n_cpus

        for _ in range(n_cpus):
            new_frame = df.sample(split_at)
            sets.append(new_frame)
            df = df.drop(new_frame.index)

        return sets, split_at, n_cpus

    data, chunk_size, n_cpus = split_jobs(dataset)

    with Pool(processes=n_cpus) as p:
        res = p.map(fn, data)  # , chunksize=chunk_size)

    return res


def freeze(m: torch.nn.Module):
    m.eval()
    for p in m.parameters():
        p.requires_grad = False


def count_params(m: torch.nn.Module):
    return {
        'requires_grad':
        sum([p.numel() for p in m.parameters() if p.requires_grad]),
        'total_params':
        sum([p.numel() for p in m.parameters()])
    }
