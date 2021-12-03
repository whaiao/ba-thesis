from typing import Callable, Iterable
from multiprocess.pool import Pool, AsyncResult

import numpy as np
import jax.numpy as jnp
import torch


def return_tensor(x: np.ndarray, return_as: str, dtype=None) -> Optional[np.ndarray, jnp.ndarray, torch.Tensor]:
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
