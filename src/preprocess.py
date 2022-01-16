import pickle
import re
from typing import List
from numpy import ndarray

import pandas as pd
import torch
from torch import Tensor

from src.eval import get_bert_score
from src.nlp import srl, dependency_parse
from src.utils import read_tsv

Dataframe = pd.DataFrame


def extract_phrases(srl_parses: dict) -> List[str]:
    def process(phrase: str) -> str:
        matches = ' '.join(re.findall(r'\[\w+:[\s\w+]+\]', phrase, re.IGNORECASE))
        matches = re.sub(r'\[\w+:\s', '', matches)
        matches = matches.replace(']', '')

        return matches

    phrases = []
    for k, v in srl_parses.items():
        for parses in v:
            phrases.append(process(parses.parse))
        
    return phrases
    
def prepare_overlap(x: str, overlap_mode: str, atomic: Dataframe,
                    dependency_parser, srl_model) -> ndarray:
    sentence = x

if __name__ == "__main__":
    with open('srl.pickle', 'rb') as p:
        data = pickle.load(p)

    extract_phrases(data)
