from dataclasses import dataclass
from pprint import pprint

import torch
from torch import Tensor
from torch import nn
from transformers import AutoTokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.models.dialog_guiding_module.dialog_guiding_module import DialogGuidingModule
from src.models.dialog_transformer import DialogTransformer
from src.utils import init_from_checkpoint, freeze_weights


@dataclass
class ModelConfig:
    device: torch.device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    d_model: int = 768

    # history transformer
    n_enc_heads: int = 8
    n_dec_heads: int = 8
    n_enc_layers: int = 2
    n_dec_layers: int = 8

    # dialog guiding module
    output_dimensions: int = 768
    soc_chem_checkpoint: str = 'checkpoints/rot_checkpoint'
    hf_checkpoint: str = 'benjaminbeilharz/bert-base-uncased-next-turn-classifier'

    # language model head
    lm_checkpoint: str = 't5-base'
    pt_checkpoint: str = 'checkpoints/t5_generator.pt'
    resume_training: bool = True


class NeuralEmpathy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(NeuralEmpathy, self).__init__()
        self.cfg = cfg
        self.dialog_transformer = DialogTransformer(
            d_model=self.cfg.d_model,
            n_enc_heads=self.cfg.n_enc_heads,
            n_enc_layers=self.cfg.n_enc_layers,
            n_dec_heads=self.cfg.n_dec_heads,
            n_dec_layers=self.cfg.n_dec_layers)

        self.dialog_guiding_module = DialogGuidingModule(
            d_model=self.cfg.d_model,
            output_dimensions=self.cfg.output_dimensions,
            soc_chem_checkpoint=self.cfg.soc_chem_checkpoint,
            hf_checkpoint=self.cfg.hf_checkpoint)

        self.lm_tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.lm_checkpoint)
        self.lm_head = T5ForConditionalGeneration.from_pretrained(
            self.cfg.lm_checkpoint)
        if self.cfg.resume_training:
            pass

    def _freeze_params(self, module_name: str):
        """Freezes weights of supplied module name

        Args:
            module_name - module to freeze
        """
        if hasattr(self, module_name):
            freeze_weights(getattr(self, module_name))

    def _unfreeze_params(self, module_name: str):
        """Unfreezes weights of supplied module name

        Args:
            module_name - module to unfreeze
        """
        if hasattr(self, module_name):
            module = getattr(self, module_name)
            module.train()
            for p in module.parameters():
                p.requires_grad = True

    def _prepare_t5_input(self, next_turn: str) -> Tensor:
        labels = self.lm_tokenizer(next_turn,
                                   padding='longest',
                                   max_length=128,
                                   truncation=True,
                                   return_tensors='pt').input_ids
        labels[labels == self.lm_tokenizer.pad_token_id] = -100
        return labels

    def forward(self, history: str, turn: str, next: str) -> Seq2SeqLMOutput:
        encoded_history = self.dialog_transformer(history, turn)
        knowledge_encoding = self.dialog_guiding_module(encoded_history, turn)
        next_utterance = self._prepare_t5_input(next)
        out = self.lm_head(inputs_embeds=knowledge_encoding,
                           labels=next_utterance)
        return out


if __name__ == "__main__":
    cfg = ModelConfig(
        soc_chem_checkpoint='src/models/social-chemistry-101/rot_checkpoint',
        resume_training=False)
    model = NeuralEmpathy(cfg=cfg)

    hist = 'how do you feel this evening?'
    query = 'i feel like i am dying'
    nxt = 'this is sad to hear, are you sure you don\'t to get help'

    out = model(hist, query, nxt)
    pprint(out)
