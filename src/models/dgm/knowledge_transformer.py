from copy import deepcopy
from math import sqrt
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from transformers import AutoModel, AutoTokenizer


class KnowledgeAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 n_context_heads: int,
                 n_mental_heads: int,
                 n_event_heads: int,
                 dropout: float = .1,
                 share_weights: bool = False):
        """Knowledge Attention Module incooperating external knowledge
        Args:
            embed_dim - model dimension (use same as knowledge encoder)
            n_context_heads - number of attention heads using for current turn
            n_mental_heads - number of attention heads using for mental state
            n_event_heads - number of attention heads using for event state
            dropout - dropout to apply on attention
            share_weights - if true uses the same linaer layer to upscale inputs
        """

        super(KnowledgeAttention, self).__init__()
        self.d_model = embed_dim
        self.n_context_heads = n_context_heads
        self.n_event_heads = n_event_heads
        self.n_mental_heads = n_mental_heads
        self.total_heads = n_context_heads + n_event_heads + n_mental_heads
        assert self.d_model % self.total_heads == 0, 'Model dimensions are not divisable through the number of heads'

        self.dropout = dropout
        self.share_weights = share_weights

        # use the same weights to encode matrices
        if self.share_weights:
            self.linear = nn.Linear(self.d_model, self.d_model * 3)
        else:
            self.context_linear = nn.Linear(self.d_model, self.d_model * 3)
            self.event_linear = deepcopy(self.context_linear)
            self.mental_linear = deepcopy(self.context_linear)

        self.output = nn.LazyLinear(self.d_model)

    def _multihead_attention(self, q: torch.FloatTensor, k: torch.FloatTensor,
                             v: torch.FloatTensor, mask: torch.BoolTensor,
                             dropout: float) -> Tuple[torch.FloatTensor]:
        """Calculates multihead attention

        Args:
            q - query matrix
            k - key matrix
            v - value matrix
            mask - attention mask
            dropout - dropout rate

        Returns:
            attention_score and attention
        """
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        if dropout > 0.0:
            attention = F.dropout(attention, p=dropout)
        output = torch.matmul(attention, v)
        return (output, attention)

    def _process_attention_heads(
            self, x: torch.FloatTensor, qkv: torch.FloatTensor, n_heads: int,
            mask: torch.BoolTensor) -> Tuple[torch.FloatTensor]:
        """Processes matrices to output attention values and weights

        Args:
            x - input sample
            qkv - upscaled weight matrix from input
            n_heads - number of attention heads
            mask - attention mask

        Returns:
            attention values and weights
        """
        batch_size, seq_len, embed_dim = x.size()

        # must calculate attention head dimension dynamically
        qkv = qkv.reshape(batch_size, seq_len, n_heads,
                          self.d_model // n_heads * 3)
        qkv = qkv.permute(0, 2, 1, 3)  # batch, heads, seqlen, dim
        q, k, v = qkv.chunk(3, dim=-1)
        output, attn = self._multihead_attention(q, k, v, mask, self.dropout)
        output = output.permute(0, 2, 1, 3)  # batch, seqlen, heads, dim
        attention_logits = output.reshape(batch_size, seq_len, embed_dim)
        # z0 x w0
        output = self.output(attention_logits)
        return (output, attn)

    def forward(self,
                context: torch.FloatTensor,
                event: torch.FloatTensor,
                mental: torch.FloatTensor,
                mask=None,
                return_weights: bool = False) -> Tuple[torch.FloatTensor]:
        """Knowledge attention forward pass

        Args:
            context - current turn
            event - event encoding
            mental - mental state encoding
            mask - attention mask
            return_weights - returns attention weights

        Returns:
            if `return_weights` is `False`: (context_attention, event_attention, mental_attention)
            else: (context_attention, event_attention, mental_attention, context_weights, event_weights, mental_weights)
        """
        if self.share_weights:
            qkv_context = self.linear(context)
            qkv_event = self.linear(event)
            qkv_mental = self.linear(mental)
        else:
            qkv_context = self.context_linear(context)
            qkv_event = self.event_linear(event)
            qkv_mental = self.mental_linear(mental)

        context_o, context_attn = self._process_attention_heads(
            context, qkv_context, self.n_context_heads, mask)
        event_o, event_attn = self._process_attention_heads(
            event, qkv_event, self.n_event_heads, mask)
        mental_o, mental_attn = self._process_attention_heads(
            mental, qkv_mental, self.n_mental_heads, mask)

        # TODO: concat attention heads and project with a linear layer into one matrix
        # fix attention mask with regards to different sized input, might as well take attention masks from tokenizer for knowledge

        # concat all transformed attention heads and feed through linear layer
        attention = torch.cat([context_o, event_o, mental_o], dim=1)
        out = self.output(attention)

        if return_weights:
            return (context_o, event_o, mental_o, context_attn, event_attn,
                    mental_attn)
        else:
            return (context_o, event_o, mental_o)


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


# testing area
if __name__ == "__main__":
    module = AtomicMultiHeadAttention(embed_dim=768,
                                      num_heads=8,
                                      dropout=0.1,
                                      checkpoint='distilbert-base-uncased')
    mha = KnowledgeAttention(768, 2, 3, 3, share_weights=False)
    x = torch.randn((2, 40, 768))
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    encode = lambda x: tokenizer(
        x, truncation=True, padding='max_length', return_tensors='pt')
    event = ['this is a sample', 'and another']
    mental = ['i am feeling well', 'so well']

    e = model(**encode(event)).last_hidden_state
    m = model(**encode(mental)).last_hidden_state

    # print(mha(x, e, m))
    mha(x, e, m)
    # print(module(x, event, mental))
