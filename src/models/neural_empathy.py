from dataclasses import dataclass
from pprint import pprint

import torch
from torch import Tensor
from torch import nn
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, AutoModelForCausalLM
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.models.dialog_guiding_module.dialog_guiding_module import DialogGuidingModule
from src.models.dialog_transformer import DialogTransformer
from src.utils import freeze_weights


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
    lm_checkpoint: str = 'benjaminbeilharz/t5-conditioned-next-turn'
    pt_checkpoint: str = 'checkpoints/t5_generator.pt'
    resume_training: bool = True


class NeuralEmpathy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(NeuralEmpathy, self).__init__()
        self.cfg = cfg

        #self.dialog_transformer = DialogTransformer(
        #    d_model=self.cfg.d_model,
        #    n_enc_heads=self.cfg.n_enc_heads,
        #    n_enc_layers=self.cfg.n_enc_layers,
        #    n_dec_heads=self.cfg.n_dec_heads,
        #    n_dec_layers=self.cfg.n_dec_layers).to(self.cfg.device)

        self.dialog_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.dialog_transformer = AutoModel.from_pretrained('bert-base-uncased').to(self.cfg.device)

        self.dialog_guiding_module = DialogGuidingModule(
            d_model=self.cfg.d_model,
            output_dimensions=self.cfg.output_dimensions,
            soc_chem_checkpoint=self.cfg.soc_chem_checkpoint,
            hf_checkpoint=self.cfg.hf_checkpoint).to(self.cfg.device)


        if 't5' in self.cfg.lm_checkpoint:
            self.lm_head = T5ForConditionalGeneration.from_pretrained(self.cfg.lm_checkpoint).to(self.cfg.device)
            self.lm_tokenizer = AutoTokenizer.from_pretrained('t5-base')
        elif 'DialoGPT' in self.cfg.lm_checkpoint:
            self.lm_head = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium').to(self.cfg.device)
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

    def _prepare_lm_input(self, next_turn: str) -> Tensor:
        labels = self.lm_tokenizer(next_turn,
                                   padding='longest',
                                   max_length=128,
                                   truncation=True,
                                   return_tensors='pt').input_ids
        labels[labels == self.lm_tokenizer.pad_token_id] = -100
        return labels.to(self.cfg.device)

    def inference(self, history: str, turn: str, **generation_settings):
        """Inference step to generate a response

        Args:
            history - dialog history
            turn - current input
            generation_settings - generation strategy to let language model generate
        
        Returns:
            a natural language response
        """
        if hasattr(self, 'dialog_tokenizer'):
            tokenized = self.dialog_tokenizer(history, 
                    turn, 
                    truncation=True, 
                    padding='max_length', 
                    return_tensors='pt').to(self.cfg.device)
            encoded_history = self.dialog_transformer(**tokenized).last_hidden_state
        else:
            encoded_history = self.dialog_transformer(history, turn)
        knowledge_encoding = self.dialog_guiding_module(encoded_history, turn)
        if 't5' in self.cfg.lm_checkpoint:
            next_utterance = self._prepare_lm_input(next)
        else:
            next_utterance = knowledge_encoding
         
        knowledge_encoding2tokens = torch.argmax(knowledge_encoding, dim=-1)

        # if settings are not supplied, use default arguments to generate
        if generation_settings is None:
            outputs = model.generate(input_ids=knowledge_encoding2tokens)
        else:
            outputs = model.generate(input_ids=knowledge_encoding2tokens, **generation_settings)

        generated = self.lm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated


    def forward(self, history: str, turn: str, next: str) -> Seq2SeqLMOutput:
        """Forward pass
        
        Args:
            history - dialog history
            turn - current utterance
            next - gold label for response to current utterance

        Returns:
            logits, loss in `Seq2SeqLMOutput`
        """
        if hasattr(self, 'dialog_tokenizer'):
            tokenized = self.dialog_tokenizer(history, turn, truncation=True, padding='max_length', return_tensors='pt').to(self.cfg.device)
            encoded_history = self.dialog_transformer(**tokenized).last_hidden_state
        else:
            encoded_history = self.dialog_transformer(history, turn)
        knowledge_encoding = self.dialog_guiding_module(encoded_history, turn)
        if 't5' in self.cfg.lm_checkpoint:
            next_utterance = self._prepare_lm_input(next)
        else:
            next_utterance = knowledge_encoding
         
        out = self.lm_head(inputs_embeds=knowledge_encoding,
                           labels=next_utterance)
        return out

