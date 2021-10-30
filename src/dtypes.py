from typing import *  # pylint-ignore-wildcard-import

import haiku as hk
import jax.numpy as jnp
from jax.random import PRNGKey


class TransformerEncoder(NamedTuple):
    module: hk.Module
    nlayers: int

class TransformerDecoder(NamedTuple):
    module: hk.Module
    nlayers: int
    output_size: int

class TransformerModel(NamedTuple):
    encoder: TransformerEncoder
    decoder: TransformerDecoder

class Session(NamedTuple):
    name: str
    cfg: Mapping[str, Any]


Function = Callable
ModuleName = Optional[str]
Data = Mapping[str, jnp.ndarray]
Sentence = List[str]
Beam = Tuple[jnp.ndarray, jnp.ndarray]
Jndarray = jnp.ndarray
RNGKey = PRNGKey

