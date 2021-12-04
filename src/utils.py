#! /usr/bin/env python3

"""
Configuration handling for experiments
"""
from pathlib import Path
from typing import Union, Mapping, Callable, Iterable, TypeVar

import jax.numpy as jnp
import numpy as np
import torch
from multiprocess.pool import Pool, AsyncResult

import yaml

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
