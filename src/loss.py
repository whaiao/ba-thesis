from typing import Mapping

import jax
import jax.numpy as jnp
import haiku as hk
from .dtypes import PRNGKey, Data

def lm_with_mask_loss_fn(forward_fn,
                         vocab_size: int,
                         params: hk.Params,
                         rng: PRNGKey,
                         data: Data,
                         is_training: bool = True) -> jnp.ndarray:
    """Cross-entropy loss with masking for Transformers

    Args:
        forward_fn: forward pass function
        vocab_size: vocab size of dataset
        params: network parameters retrieved via `init`
        rng: `PRNGKey` from haiku
        data: dataset
        is_training: true (default)

    Returns:
        cross-entropy loss with masking function
    """

    logits: jnp.ndarray = forward_fn(params, rng, data, is_training)
    targets: jnp.ndarray = jax.nn.one_hot(data['target'], vocab_size)
    assert logits.shape == targets.shape

    mask: jnp.ndarray = jnp.greater(data['obs'], 0)
    loss: jnp.ndarray = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    loss: jnp.ndarray = jnp.sum(loss * mask) / jnp.sum(mask)

    return loss