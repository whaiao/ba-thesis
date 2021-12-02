from pprint import pprint
from typing import Tuple, Union

import datasets
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

Pipeline = Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]
TorchType = torch.dtype
NumpyType = np.dtype
JaxType = jax.numpy.dtype
Datatypes = Union[TorchType, NumpyType, JaxType]


def init_huggingface_models(model: str) -> Pipeline:
    tokenizer = transformers.BertTokenizer.from_pretrained(model)
    model = transformers.BertModel.from_pretrained(model)
    return (model, tokenizer)


def forward(x: str, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, tensor_type: str):
    embedding = tokenizer(x, return_tensors=tensor_type)
    output = model(**embedding)
    return embedding, output


def ed():
    model, tokenizer = init_huggingface_models('bert-base-uncased')
    data = datasets.load_dataset('empathetic_dialogues')['train']['utterance']
    emb, out = forward(data[0], model, tokenizer, tensor_type='pt')
    return out, emb


def test():
    model, tokenizer = init_huggingface_models('bert-base-uncased')
    tokenized, out = forward('This is a test string', model, tokenizer, tensor_type='pt')
    pprint(tokenized)
    # pprint(out)

test()

