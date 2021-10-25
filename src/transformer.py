from .dtypes import *

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

class SelfAttention(hk.MultiHeadAttention):
    def __call__(self,
                 query: jnp.ndarray,
                 key: Optional[jnp.ndarray] = None,
                 value: Optional[jnp.ndarray] = None,
                 mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        key = key if key is not None else query
        value = value if value is not None else query

        seq_len = query.shape[1]
        c_mask = np.tril(np.ones((seq_len, seq_len)))
        mask = mask * c_mask if mask is not None else c_mask
        return super().__call__(query, key, value, mask)


class Linear(hk.Module):
    def __init__(self, init_scale: float,
                 widening_factor: int = 4,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        init = hk.initializers.VarianceScaling(self._init_scale)
        x = hk.Linear(self._widening_factor*hiddens, w_init=init)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(hiddens, w_init=init)(x)

def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    return hk.LayerNorm(axis=-1,
                        create_scale=True,
                        create_offset=True,
                        name=name)(x)


class Transformer(hk.Module):

    def __init__(self, nheads: int,
                 nlayers: int,
                 dropout_rate: float,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._nlayers = nlayers
        self._nheads = nheads
        self._dropout_rate = dropout_rate

    def __call__(self,
                 h: jnp.ndarray,
                 mask: Optional[jnp.ndarray],
                 is_training: bool) -> jnp.ndarray:
        init_scale = 2. / self._nlayers
        dropout_rate = self._dropout_rate if is_training else 0.
        if mask is not None:
            mask = mask[:, None, None, :]

        for i in range(self._nlayers):
            h_norm = layer_norm(h, name=f'h{i}_ln_1')
            h_attn = SelfAttention(
                num_heads=self._nheads,
                key_size=64,
                w_init_scale=init_scale,
                name=f'h{i}_attn')(h_norm, mask=mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h += h_attn
            h_norm = layer_norm(h, name=f'h{i}_ln_2')
            h_dense = Linear(init_scale, name=f'h{i}_linear')(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h += h_dense
        h = layer_norm(h, name='ln_f')
        return h


def embeddings(data: Mapping[str, jnp.ndarray], vocab_size: int) -> jnp.ndarray:
    tokens = data  # TODO: define input here
    input_mask = jnp.greater(tokens, 0)
    seq_len = tokens.shape[1]

    embed_init = hk.initializers.TruncatedNormal(stddev=.02)
    token_embedding_map = hk.Embed(vocab_size, d_model, w_init=embed_init)
    token_embs = token_embedding_map(tokens)
    positional_embeddings = hk.get_parameter('pos_embs', [seq_len, d_model], init=embed_init)
    input_embeddings = token_embs + positional_embeddings
    return input_embeddings, input_mask

def build_transformer_forward_fn(vocab_size: int, d_model: int, nheads: int, nlayers: int, dropout_rate: float):
    def forward_fn(data: Data, is_training: bool = True) -> jnp.ndarray:
        input_embeddings, input_mask = embeddings(data, vocab_size)
        transformer = Transformer(nheads=nheads, nlayers=nlayers, dropout_rate=dropout_rate)
        output_embeddings = transformer(input_embeddings, input_mask, is_training)
        return hk.Linear(vocab_size)(output_embeddings)

    return forward_fn