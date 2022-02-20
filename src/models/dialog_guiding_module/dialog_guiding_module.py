from pprint import pprint
from typing import Iterable, List, Mapping, Tuple

from src.knowledge_extraction import extract_from_atomic, retrieve_overlap
from torch import Tensor
from torch import nn
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.models.dialog_guiding_module.knowledge_transformer import KnowledgeAttention, KnowledgeAttentionEncoder
from src.models.dialog_transformer import DialogTransformer


class DialogGuidingModule(nn.Module):
    def __init__(self,
                 d_model: int = 768,
                 output_dimensions: int = 512,
                 hf_checkpoint: str = 'distilbert-base-uncased'):
        """DialogGuidingModule which extracts knowledge from Atomic, predicts next turn type
        and encodes knowledge via attention heads pointing to pre-Language Model encoder

        Args:
            d_model - embedding dimensions
            output_dimensions - outform transformation for language model head capability
            hf_checkpoint - huggingface checkpoint for tokenizer
        """

        super(DialogGuidingModule, self).__init__()
        #self.next_turn_predictor = AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)
        self.knowledge_attention = KnowledgeAttention(d_model, 4, 4, 4, 4)
        self.knowledge_encoder = KnowledgeAttentionEncoder()
        # prepare input for specific language model head
        self.projection_layer = nn.Linear(d_model, output_dimensions)

    def _knowledge_lookup(
            self,
            query: str) -> Iterable[Mapping[str, Mapping[str, List[str]]]]:
        """Receives knowledge from Atomic graph
        
        Args:
            query - query to search

        Returns:
            knowledge mappings
        """

        return extract_from_atomic(retrieve_overlap(query))

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

        mental = ''
        event = ''
        moral = ''

        def extract(tails: Iterable[str]) -> str:
            return ' '.join(tails)

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

        # TODO: add string repr of next turn prediction to knowledge encoder
        encoded_knowledge = self.knowledge_encoder(x, knowledge)
        out = self.projection_layer(encoded_knowledge)
        return out


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained('t5-small')
    t5 = T5ForConditionalGeneration.from_pretrained('t5-small')
    dialog_transformer = DialogTransformer(768)
    model = DialogGuidingModule()
    hist = 'how do you feel this evening?'
    query = 'i feel like i am dying'
    nxt = 'this is sad to hear, are you sure you don\'t to get help'
    x = dialog_transformer(hist, query)
    out = model(x, query)
    nxt = tok(nxt, truncation=True, padding='max_length',
              return_tensors='pt').input_ids
    pprint(t5(inputs_embeds=out, labels=nxt))
