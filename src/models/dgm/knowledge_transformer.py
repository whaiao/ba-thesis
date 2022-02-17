from copy import deepcopy
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from transformers import AutoModel, AutoTokenizer


class AtomicMultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 checkpoint: str,
                 batch_first: bool = True,
                 share_weights: bool = True):
        """Multihead Attention Module for Knowledge

        Args:
            embed_dim - model hidden dimension
            num_heads - number of heads to use for each source
            dropout - dropout ratio
            checkpoint - huggingface checkpoint for knowledge string encoding
            batch_first - batch_size first -> (batch_size x sequence_length x hidden_dimension)
            share_weights - if true share weights between knowledge encoders

        """
        super(AtomicMultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, 'Embedding dimension is not divisable through number of attention heads'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = True
        self.checkpoint = 'distilbert-base-uncased' if checkpoint is None else checkpoint
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.encoder = AutoModel.from_pretrained(self.checkpoint)
        self.encoder_ff = nn.LazyLinear(embed_dim, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.share_weights = share_weights

        if share_weights:
            self.mental_encoder = deepcopy(self.encoder)
            self.mental_encoder_ff = nn.LazyLinear(embed_dim,
                                                   device=self.device)
            self.event_encoder = deepcopy(self.encoder)
            self.event_encoder_ff = nn.LazyLinear(embed_dim,
                                                  device=self.device)
            del self.encoder

        self.context_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=batch_first,
            device=self.device)

        self.mental_attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                                      num_heads=self.num_heads,
                                                      dropout=dropout,
                                                      batch_first=batch_first,
                                                      device=self.device)

        self.event_attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                                     num_heads=self.num_heads,
                                                     dropout=dropout,
                                                     batch_first=batch_first,
                                                     device=self.device)

    # implementation only for batch_size 1
    def embed(self, x: str, encoder_type: str) -> torch.FloatTensor:
        """Creates embeddings of atomic relation inputs and fix them onto a given size

        Args:
            x - input sequence
            encoder_type - type of encoder to use if multiple encoders are supplied
        Returns:
            Embedding of size (batch_size [1], hidden_dim [256])
        """

        if not self.share_weights:
            inputs = self.tokenizer(x, truncation=True, return_tensors='pt')
            embedding = self.encoder(**inputs).last_hidden_state
            embedding = embedding.view(1, -1)
            # (batch_size, seq_len, embedding_dim)
            out = self.encoder_ff(embedding)
            return out

        assert encoder_type in ['mental', 'event']
        if encoder_type == 'mental':
            inputs = self.tokenizer(x,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt')
            embedding = self.mental_encoder(**inputs).last_hidden_state
            return embedding

        if encoder_type == 'event':
            inputs = self.tokenizer(x,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt')
            embedding = self.event_encoder(**inputs).last_hidden_state
            return embedding

    def forward(self,
                x: torch.FloatTensor,
                event: str,
                mental: str,
                return_weights: bool = False) -> Tuple[torch.FloatTensor]:
        """Forward function for multihead knowledge attention

        Args:
            x - embedded input tensor of current utterance
            event - atomic event relations as a concatenated string
            mental - atomic mental relations as a concatenated string
            return_weights - return attenion weights if `True`

        Returns:
            if `return_weights` is `False`: (context_attention, event_attention, mental_attention)
            else: (context_attention, event_attention, mental_attention, context_weights, event_weights, mental_weights)
        """

        if self.share_weights:
            event_emb = self.embed(event, 'event')
            mental_emb = self.embed(mental, 'mental')
        else:
            event_emb = self.embed(event, None)
            mental_emb = self.embed(mental, None)

        event_attn, event_weights = self.event_attention(query=x,
                                                         key=event_emb,
                                                         value=event_emb)
        mental_attn, mental_weights = self.mental_attention(query=x,
                                                            key=mental_emb,
                                                            value=mental_emb)
        context_attn, context_weights = self.context_attention(
            x,
            x,
            x,
        )
        if not return_weights:
            return (context_attn, event_attn, mental_attn)

        return (context_attn, event_attn, mental_attn, context_weights,
                event_weights, mental_weights)


if __name__ == "__main__":
    # testing area
    module = AtomicMultiHeadAttention(embed_dim=768,
                                      num_heads=8,
                                      dropout=0.1,
                                      checkpoint='distilbert-base-uncased')
    x = torch.randn((2, 40, 768))
    event = ['this is a sample', 'and another']
    mental = ['i am feeling well', 'so well']

    print(module(x, event, mental))
