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
