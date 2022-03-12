from pprint import pprint
from typing import Iterable, List, Mapping, Tuple, Union

# hy needed for modules written in hy-lang [knowledge_extraction]
import hy
import torch
from torch import Tensor
from torch import nn
from transformers import AutoTokenizer, AutoModelWithLMHead, T5ForConditionalGeneration, AutoModelForSequenceClassification

from src.constants import T5_TURN_TEMPLATES
from src.knowledge_extraction import extract_from_atomic, retrieve_overlap
from src.models.dialog_guiding_module.knowledge_transformer import KnowledgeAttention, KnowledgeAttentionEncoder
from src.models.dialog_transformer import DialogTransformer
from src.utils import freeze_weights


class DialogGuidingModule(nn.Module):
    def __init__(self,
                 d_model: int = 768,
                 output_dimensions: int = 512,
                 soc_chem_checkpoint:
                 str = 'src/models/social-chemistry-101/rot_checkpoint',
                 hf_checkpoint: str = 'distilbert-base-uncased',
                 device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        """DialogGuidingModule which extracts knowledge from Atomic, predicts next turn type
        and encodes knowledge via attention heads pointing to pre-Language Model encoder

        Args:
            d_model - embedding dimensions
            output_dimensions - outform transformation for language model head capability
            hf_checkpoint - huggingface checkpoint for tokenizer
        """

        super(DialogGuidingModule, self).__init__()

        # predict next turn and prepend template
        self.device = device
        self.templates = T5_TURN_TEMPLATES
        self.next_turn_predictor = AutoModelForSequenceClassification.from_pretrained(
            hf_checkpoint)
        freeze_weights(self.next_turn_predictor)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)

        # used for social chemistry encoder model
        self.moral_tokenizer = AutoTokenizer.from_pretrained(
            soc_chem_checkpoint, model_max_length=512)

        self.moral_tokenizer.pad_token = self.moral_tokenizer.eos_token
        self.moral_gpt = AutoModelWithLMHead.from_pretrained(
            soc_chem_checkpoint).to(self.device)
        self.moral_gpt_out = nn.Sequential(nn.Linear(1600, d_model), nn.ReLU())
        freeze_weights(self.moral_gpt)

        self.moral_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        # ---

        self.knowledge_attention = KnowledgeAttention(d_model, 4, 4, 4, 4)
        self.knowledge_encoder = KnowledgeAttentionEncoder()
        # prepare input for specific language model head
        self.projection_layer = nn.Linear(d_model, output_dimensions)

    def _classify_next_turn_type(
        self, string_repr: Union[str, Iterable[str]]
    ) -> Union[Tensor, Tuple[str, Tensor]]:
        """Classify next turn type
        
        Args:
            string_repr - string or batch of string representations to classify

        Returns:
            next turn type label"""
        tokenized = self.tokenizer(string_repr,
                                   truncation=True,
                                   padding='max_length',
                                   return_tensors='pt').to(self.device)
        out = self.next_turn_predictor(**tokenized)
        logits = out.logits
        preds = torch.argmax(logits, dim=-1)
        return preds

    def _produce_moral_encoding(self,
                                string_repr: str,
                                moral_attention_head: Tensor,
                                action_type: str = None) -> Tensor:
        """Produces moral embedding using pretrained NeuralNormTransformer

        Args:
            string_repr - (batch of) string to tokenize
            moral_attention_head - moral attention head from `KnowledgeTransformer`

        Returns:
            moral attention head
        """
        def _prepare_for_label_classification(x: str = string_repr) -> Tensor:
            """Prepares input representation for specific characteristica classification

            Args:
                x - Input string (not processed)

            Returns:
                encoded input string for classification & generation
            """
            pass

        ins = self.moral_tokenizer(string_repr,
                                   truncation=True,
                                   padding='max_length',
                                   return_tensors='pt')

        ins = {k: v.to(self.device) for k, v in ins.items()}

        # do not fine-tune social-chemistry-101 gpt2
        with torch.no_grad():
            moral_logits = self.moral_gpt(
                **ins, labels=ins['input_ids'],
                output_hidden_states=True).hidden_states
            moral_logits = moral_logits[-1]

        # adding batch size
        intermediate = self.moral_gpt_out(moral_logits)
        moral_emb = torch.cat([intermediate, moral_attention_head], dim=-1)
        out = self.moral_projection(moral_emb)
        return out

    def _knowledge_lookup(
            self,
            query: str) -> Iterable[Mapping[str, Mapping[str, List[str]]]]:
        """Receives knowledge from Atomic graph
        
        Args:
            query - query to search

        Returns:
            knowledge mappings
        """
        ### EXPERIMENTAL: CAP MAX TOKENS TO 512
        ntokens = len(query.split())
        if ntokens > 512:
            query = ' '.join(query.split()[:512])

        ###

        overlaps = retrieve_overlap(query)
        if overlaps is None:
            return None

        return extract_from_atomic(overlaps)

    def _prepare_relations(
        self, samples: Iterable[Mapping[str,
                                        Mapping[str,
                                                List[str]]]]) -> Tuple[str]:
        """Extracts all the information from retrieved Atomic graph and prepares strings

        Args:
            samples - samples to extract knowledge from

        Returns:
            knowledge encoded strings to be encoded
        """
        if samples is None:
            return ('none', 'none', 'none')

        mental = ''
        event = ''
        moral = ''

        def extract(tails: Iterable[str]) -> str:
            tails = list(filter(lambda x: isinstance(x, str), tails))
            return ' '.join((t for t in tails if t != 'none'))

        # check if samples acutally have a sample

        for sample in samples:
            for entry in sample.values():
                for k, v in entry.items():
                    # mental relations
                    if k.startswith('x'):
                        mental += f' [{k}] '
                        mental += extract(v)
                    # moral relations
                    elif k.startswith('o'):
                        moral += f' [{k}] '
                        moral += extract(v)
                    # event relations
                    elif k in [
                            'HinderedBy', 'isAfter', 'isBefore', 'Causes',
                            'HasSubevent'
                    ]:
                        event += f' [{k}] '
                        event += extract(v)

        return (event.strip(), mental.strip(), moral.strip())

    def parse(self, string_repr: str) -> Tuple[str]:
        """Extract and prepare

        Args:
            string_repr - string representation of current input turn
        """

        s = self._knowledge_lookup(string_repr)
        return self._prepare_relations(s)

    def forward(self, x: Tensor, string_repr: str):
        """Forward pass through `DialogGuidingModule`

        Args:
            x - input representation of `DialogTransformer`
            string_repr - string representation of current utterance

        Returns:
            encoded representation for language model head
        """
        event, mental, moral = self.parse(string_repr)
        knowledge = self.knowledge_attention(x,
                                             event=event,
                                             mental=mental,
                                             moral=moral)

        moral_head = knowledge['moral']
        moral = self._produce_moral_encoding(string_repr,
                                             moral_attention_head=moral_head)
        knowledge['moral'] = moral

        # add batch encoding
        next_turn_type = int(
            self._classify_next_turn_type(string_repr).cpu().detach().numpy())
        new_representation = self.templates[next_turn_type] + string_repr
        encoded_knowledge = self.knowledge_encoder(new_representation,
                                                   list(knowledge.values()))

        out = self.projection_layer(encoded_knowledge)
        return out


if __name__ == "__main__":
    pass
    #tok = AutoTokenizer.from_pretrained('t5-base')
    ## small = 512
    ## base = 768 embedding size
    #t5 = T5ForConditionalGeneration.from_pretrained('t5-base')
    #dialog_transformer = DialogTransformer(768)
    #model = DialogGuidingModule(
    #    output_dimensions=768,
    #    hf_checkpoint='benjaminbeilharz/bert-base-uncased-next-turn-classifier'
    #)
    #hist = 'how do you feel this evening?'
    #query = 'i feel like i am dying'
    #nxt = 'this is sad to hear, are you sure you don\'t to get help'
    #x = dialog_transformer(hist, query)
    #out = model(x, query)
    #nxt = tok(nxt, truncation=True, padding='max_length',
    #          return_tensors='pt').input_ids
    #out = t5(inputs_embeds=out, labels=nxt)
    #print(out)
