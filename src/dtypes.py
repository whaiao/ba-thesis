from typing import *  # pylint-ignore-wildcard-import
import jax.numpy as jnp
from jax.random import PRNGKey
import haiku as hk


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


ModuleName = Optional[str]
Data = Mapping[str, jnp.ndarray]
Sentence = List[str]
Beam = Tuple[jnp.ndarray, jnp.ndarray]
Jndarray = jnp.ndarray
RNGKey = PRNGKey

