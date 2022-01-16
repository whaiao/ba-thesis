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


def extract_phrases(srl_parses: List[dict]) -> List[str]:
    def process(phrase: str) -> str:
        matches = re.findall(r'\[\w+:[\s\w+]+\]', phrase, re.IGNORECASE)
        for i in matches:
            i = re.sub(r'\[\w+:\s', '', i)
            i = i.replace('[', '')

        return ' '.join(matches)


def prepare_overlap(x: str, overlap_mode: str, atomic: Dataframe,
                    dependency_parser, srl_model) -> ndarray:
    sentence = x
