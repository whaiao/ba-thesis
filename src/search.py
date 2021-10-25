from .dtypes import *
import jax.numpy as jnp


def beam_search(model: TransformerModel, size: int, encoder_output: Jndarray,
        s_mask: Jndarray, max_output_length: int, alpha: float, n_best: int = 1) -> Beam:

    """Beam search with size k for Transformer.
    Find the k most likely hypotheses in each decoding step.

    Args:
        model - Transformer model
        size - Beam size
        encoder_output - Potentials
        s_mask - Source input mask
        max_output_length - 
        alpha - Length penality
        n_best - Number of hypotheses to return

    Returns:
        Beam - tuple of stacked output hypotheses (2D array of indices) & attention scores (3D array)
    """

    assert size > 0, 'Beam size must be >0.'
    assert n_best <= size, f'Can only return {size} best hypotheses'


    t_vocab_size = model.decoder.output_size
    transformer = isinstance(model.decoder, TransformerDecoder)
    batch_size = s_mask.shape[0]
    t_mask = None


    pass


