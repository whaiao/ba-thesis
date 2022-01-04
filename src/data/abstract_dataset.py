"""
Declares abstract dataset class and implements datasets used in the project
"""
from abc import ABC, abstractmethod
from typing import List, Union

import jax.numpy as jnp
import numpy as np

TorchType = torch.dtype
NumpyType = np.dtype
JaxType = jnp.dtype
Datatypes = Union[TorchType, NumpyType, JaxType]


class Dataset(ABC):
    @property
    @abstractmethod
    def headers(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def train(self):
        return self._train

    @property
    @abstractmethod
    def valid(self):
        return self._valid

    @property
    @abstractmethod
    def test(self):
        return self._test

    @classmethod
    @abstractmethod
    def load(self, name: Union[None, str]):
        raise NotImplementedError('Implement a load function for this dataset')
        

    @abstractmethod
    def save(self):
        raise NotImplementedError('Implement a save function for this dataset')

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def prepare_tensors(self, tensor_type: str, dtype: Datatypes):
        if tensor_type == 'torch':
            return (torch.tensor(self.train, dtype), torch.tensor(self.valid, dtype), torch.tensor(self.test, dtype))
        elif tensor_type == 'numpy':
            return (np.array(self.train, dtype), np.array(self.valid, dtype), np.array(self.test, dtype))
        elif tensor_type == 'jax':
            return (np.array(self.train, dtype), np.array(self.valid, dtype), np.array(self.test, dtype))
        else:
            raise ValueError(f'{dtype} is not supported.')

