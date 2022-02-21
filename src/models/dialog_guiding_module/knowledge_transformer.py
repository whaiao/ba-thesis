from collections import OrderedDict
from copy import deepcopy
from math import sqrt
from pprint import pprint
from typing import Callable, Iterable, Tuple, Optional, Union, OrderedDict

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, T5ForConditionalGeneration, AutoTokenizer


class KnowledgeAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 n_context_heads: int,
                 n_event_heads: int,
                 n_mental_heads: int,
                 n_moral_heads: int,
                 dropout: float = .1,
                 hf_checkpoint: str = 'distilbert-base-uncased',
                 share_weights: bool = False):
        """Knowledge Attention Module incooperating external knowledge
        Args:
            embed_dim - model dimension (use same as knowledge encoder)
            n_context_heads - number of attention heads using for current turn
            n_event_heads - number of attention heads using for event state
            n_mental_heads - number of attention heads using for mental state
            n_moral_heads - number of attention heads using for moral state
            dropout - dropout to apply on attention
            share_weights - if true uses the same linaer layer to upscale inputs
        """

        super(KnowledgeAttention, self).__init__()
        self.d_model = embed_dim
        self.n_context_heads = n_context_heads
        self.n_event_heads = n_event_heads
        self.n_mental_heads = n_mental_heads
        self.n_moral_heads = n_moral_heads
        self.total_heads = n_context_heads + n_event_heads + n_mental_heads + n_moral_heads
        assert self.d_model % self.total_heads == 0, 'Model dimensions are not divisable through the number of heads'

        self.dropout = dropout
        self.share_weights = share_weights
        self.tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)
        self.encoder = AutoModel.from_pretrained(hf_checkpoint)

        # use the same weights to encode matrices
        if self.share_weights:
            self.linear = nn.Linear(self.d_model, self.d_model * 3)
            self.upscale = nn.Linear(self.d_model, self.d_model * 2)
            self.pooling = nn.AdaptiveAvgPool1d(self.d_model)
        else:
            self.context_linear = nn.Linear(self.d_model, self.d_model * 3)
            self.event_linear = deepcopy(self.context_linear)
            self.mental_linear = deepcopy(self.context_linear)
            self.moral_linear = deepcopy(self.context_linear)

            # upscale
            self.context_upscale = nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2), nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2))
            self.event_upscale = deepcopy(self.context_upscale)
            self.mental_upscale = deepcopy(self.context_upscale)
            self.moral_upscale = deepcopy(self.context_upscale)

        self.output = nn.Linear(self.d_model, self.d_model)

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

    def _prepare_knowledge(self, event: str, mental: str,
                           moral: str) -> Tuple[Tensor]:
        def encode(x: str) -> Tensor:
            return self.tokenizer(x,
                                  truncation=True,
                                  padding='max_length',
                                  return_tensors='pt')

        event = self.encoder(**encode(event)).last_hidden_state
        mental = self.encoder(**encode(mental)).last_hidden_state
        moral = self.encoder(**encode(moral)).last_hidden_state
        return (event, mental, moral)

    def forward(
        self,
        context: torch.FloatTensor,
        event: str,
        mental: str,
        moral: str,
        mask: torch.BoolTensor = None,
        return_weights: bool = False
    ) -> Union[OrderedDict[str, Tensor], Tuple[torch.FloatTensor]]:
        """Knowledge attention forward pass

        Args:
            context - current turn
            event - event encoding
            mental - mental state encoding
            moral - moral encoding
            mask - attention mask
            return_weights - returns attention weights

        Returns:
            if `return_weights` is `False`: (context_attention, event_attention, mental_attention)
            else: (context_attention, event_attention, mental_attention, context_weights, event_weights, mental_weights)
        """
        event, mental, moral = self._prepare_knowledge(event, mental, moral)
        if self.share_weights:
            qkv_context = self.linear(context)
            qkv_event = self.linear(event)
            qkv_mental = self.linear(mental)
            qkv_moral = self.linear(moral)
        else:
            qkv_context = self.context_linear(context)
            qkv_event = self.event_linear(event)
            qkv_mental = self.mental_linear(mental)
            qkv_moral = self.moral_linear(moral)

        context_o, context_attn = self._process_attention_heads(
            context, qkv_context, self.n_context_heads, mask)
        event_o, event_attn = self._process_attention_heads(
            event, qkv_event, self.n_event_heads, mask)
        mental_o, mental_attn = self._process_attention_heads(
            mental, qkv_mental, self.n_mental_heads, mask)
        moral_o, moral_attn = self._process_attention_heads(
            moral, qkv_moral, self.n_moral_heads, mask)

        # fix attention mask with regards to different sized input, might as well take attention masks from tokenizer for knowledge

        if return_weights:
            return (context_o, event_o, mental_o, moral_o, context_attn,
                    event_attn, mental_attn, moral_attn)
        else:
            # linear + relu + pooling
            context_o = self.context_upscale(context_o)
            event_o = self.context_upscale(event_o)
            mental_o = self.mental_upscale(mental_o)
            moral_o = self.moral_upscale(moral_o)
            out = OrderedDict([('context', context_o), ('moral', moral_o),
                               ('mental', mental_o), ('event', mental_o)])
            # return (context_o, event_o, mental_o, moral_o)
            return out


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


class KnowledgeEncoderBlock(nn.TransformerEncoderLayer):
    def init(self,
             d_model: int = 768,
             nhead: int = 4,
             dim_feedforward: int = 2048,
             dropout: float = .1,
             activation: Callable = nn.ReLU):
        """Overwrites Transformer Encoder Layer to adjust to receiving knowledge attention heads
        Args:
            d_model - embed into dimensions
            nhead - number of knowledge attention heads
            dim_feedforward - dimension of feed forward layer
            dropout - dropout rate
            activation - activation function to use on hidden layer
        """
        super(KnowledgeEncoderBlock,
              self).__init__(d_model=d_model,
                             nhead=nhead,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout,
                             activation=activation,
                             batch_first=True)

    def forward(self,
                src: Tensor,
                knowledge_attn_head: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Forward Layer for Knowledge Self Attention
        Args:
            src - source tensor
            knowledge_attn_head - knowledge attention to use in self attention
            src_mask - mask for src tensor
            src_key_padding_mask - key padding mask

        Returns:
            Tensor after encoding layer
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), knowledge_attn_head,
                                   src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, knowledge_attn_head, src_mask,
                                              src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x: Tensor, knowledge_attn_head: Tensor,
                  attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor]) -> Tensor:
        """Self attention block adjusted to taking additional heads from KnowledgeEncoder
        Args:
            x - input tensor
            knowledge_attn_head - output from `KnowledgeEncoder`
            attn_mask - attention mask for input tensor
            key_padding_mask - key padding mask

        Returns:
            self attention output
        """
        x = self.self_attn(x,
                           knowledge_attn_head,
                           knowledge_attn_head,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)


class KnowledgeAttentionEncoder(nn.Module):
    def __init__(self, encoder_checkpoint: str = 'distilbert-base-uncased'):
        super(KnowledgeAttentionEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_checkpoint)
        self.encoding_layers = nn.ModuleList(
            [KnowledgeEncoderBlock(d_model=768, nhead=4) for _ in range(4)])

    def forward(self, x: Union[str, Tensor],
                knowledge_attn_heads: Iterable[Tensor]) -> Tensor:

        assert len(knowledge_attn_heads) == len(
            self.encoding_layers
        ), 'Number of attention encoding layers does not match number of knowledge attention heads'
        if isinstance(x, str):
            encode = self.tokenizer(x,
                                    truncation=True,
                                    padding='max_length',
                                    return_tensors='pt')
            x = self.encoder(**encode).last_hidden_state
        elif isinstance(x, torch.FloatTensor):
            x = self.encoder(inputs_embeds=x).last_hidden_state

        for layer, knowledge_attn_head in zip(self.encoding_layers,
                                              knowledge_attn_heads):
            x = layer(src=x, knowledge_attn_head=knowledge_attn_head)
        return x


# testing area
if __name__ == "__main__":
    mha = KnowledgeAttention(768, 4, 4, 4, 4, share_weights=False)
    knowledge_encoder_block = KnowledgeEncoderBlock(d_model=768, nhead=4)
    knowledge_encoder = KnowledgeAttentionEncoder()

    # ensure inputs are fixed to 512
    x = torch.randn((2, 512, 768))
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    t5 = T5ForConditionalGeneration.from_pretrained('t5-small')
    t5_tok = AutoTokenizer.from_pretrained('t5-small')
    encode = lambda x, y: y(
        x, truncation=True, padding='max_length', return_tensors='pt')
    event = ['this is a sample', 'and another']
    mental = ['i am feeling well', 'so well']
    moral = ['i am feeling well', 'so well']

    #    e = model(**encode(event, tokenizer)).last_hidden_state
    #    m = model(**encode(mental, tokenizer)).last_hidden_state
    #    mo = model(**encode(moral, tokenizer)).last_hidden_state

    knowledge = mha(x, event, mental, moral)

    # t5 needs embedding dim of 512
    x = ['This is a sample', 'And another one']
    linear = nn.Linear(768, 512)
    t5_in = knowledge_encoder(x, knowledge)
    t5_in = linear(t5_in)

    # batch_size must be fitting
    dec = encode(['This is a sample', 'And another one'], t5_tok).input_ids
    out = t5(inputs_embeds=t5_in, labels=dec)
    pprint(out)
