from pprint import pprint
from typing import Tuple, Union, Iterable
import torch
from torch import nn
from torch import Tensor

from transformers import AutoTokenizer, PreTrainedTokenizer


class HistoryEncoder(nn.Module):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 d_model: int = 768,
                 n_heads: int = 8,
                 n_layers: int = 2):
        """Chat History Transformer Encoder

        Args:
            tokenizer - Huggingface Tokenizer
            d_model - embedding dimension
            n_heads - number of heads in `SelfAttention`
            n_layers - number of layers to stack
        """
        super(HistoryEncoder, self).__init__()
        assert d_model % n_heads == 0, 'Embedding dim not divisable through number of heads'
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=n_heads,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer,
                                             num_layers=n_layers)

    def _tokenize(self, x: Union[str, Iterable[str]]) -> Tuple[Tensor]:
        """Prepares string input to feed into Transformer
        Args:
            x - string or a list or strings

        Returns:
            input ids and attention mask
        """
        tokenized = self.tokenizer(x,
                                   truncation=True,
                                   padding='max_length',
                                   return_tensors='pt')
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask.to(torch.bool)
        return (input_ids, attention_mask)

    def forward(self, x: Union[str, Iterable[str]]) -> Tuple[Tensor]:
        """Encodes dialog history

        Args:
            x - history or list of histories

        Returns:
            hidden representation and mask
        """
        ids, mask = self._tokenize(x)
        embs = self.embedding(ids)
        out = self.encoder(src=embs, src_key_padding_mask=mask)
        return (out, mask)


class UtteranceDecoder(nn.Module):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 d_model: int = 768,
                 n_heads: int = 16,
                 n_layers: int = 8):
        """Utterance Transformer Decoder

        Args:
            tokenizer - Huggingface Tokenizer
            d_model - embedding dimension
            n_heads - number of heads in `SelfAttention`
            n_layers - number of layers to stack
        """
        super(UtteranceDecoder, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                        nhead=n_heads,
                                                        batch_first=True)

        self.decoder = nn.TransformerDecoder(self.decoder_layer,
                                             num_layers=n_layers)

    def _tokenize(self, x: Union[str, Iterable[str]]) -> Tuple[Tensor]:
        """Prepares string input to feed into Transformer
        Args:
            x - string or a list or strings

        Returns:
            input ids and attention mask
        """
        tokenized = self.tokenizer(text=x,
                                   truncation=True,
                                   padding='max_length',
                                   return_tensors='pt')
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask.to(torch.bool)

        return (input_ids, attention_mask)

    def forward(self, x: Union[str, Iterable[str]],
                history: Union[Tensor, Iterable[Tensor]],
                history_mask: Union[Tensor, Iterable[Tensor]]) -> Tensor:
        """Decodes current utterance and feed it through multiple decoder layers with attention passes 
        from the `HistoryEncoder`
        Args:
            x - utterance
            history - encoded history (output from `HistoryEncoder`)
            history_mask - masks from history encoding (output from `HistoryEncoder`)

        Returns:
            decoded tensor
        """
        ids, mask = self._tokenize(x)
        embs = self.embedding(ids)
        out = self.decoder(tgt=embs,
                           memory=history,
                           tgt_key_padding_mask=mask,
                           memory_key_padding_mask=history_mask)
        return out


class DialogTransformer(nn.Module):
    def __init__(self,
                 d_model: int = 768,
                 n_enc_heads: int = 8,
                 n_enc_layers: int = 2,
                 n_dec_heads: int = 16,
                 n_dec_layers: int = 8,
                 hf_checkpoint: str = 'distilbert-base-uncased'):
        """Builds `DialogTransformer` for Dialog History Encoding and current Utterance Processing
        
        Args:
            d_model - embedding dimension
            n_enc_heads - number of encoder attention heads
            n_enc_layers - number of encoder layers stacked on each other
            n_dec_heads - number of decoder attention heads
            n_dec_layers - number of decoder layers stacked on each other
            hf_checkpoint - tokenizer checkpoint to use
        """
        super(DialogTransformer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)
        self.encoder = HistoryEncoder(self.tokenizer,
                                      d_model=d_model,
                                      n_heads=n_enc_heads,
                                      n_layers=n_enc_layers)
        self.decoder = UtteranceDecoder(self.tokenizer,
                                        d_model=d_model,
                                        n_heads=n_dec_heads,
                                        n_layers=n_dec_layers)

    def forward(self, history: Union[str, Iterable[str]],
                utterance: Union[str, Iterable[str]]):
        enc_out, enc_mask = self.encoder(history)
        dec_out = self.decoder(x=utterance,
                               history=enc_out,
                               history_mask=enc_mask)
        return dec_out


if __name__ == "__main__":
    transformer = DialogTransformer()
    pprint(
        transformer(history=[
            'This is a sample', 'Another one',
            'Yet another one for the encoder'
        ],
                    utterance=[
                        'This should be the answer', 'This also',
                        'And this as well'
                    ]))
