
from collections import defaultdict
from datetime import date
from typing import Callable, Tuple, Mapping, NamedTuple, Set, Union, List

import pandas as pd
import spacy
import spacy.symbols as S
from spacy import displacy
from spacy.tokens.doc import Doc
from spacy.tokens.token import Token

from src.utils import read_tsv, sorted_dict
from src.data.save import to_pickle

# dtypes and constants
try:
    NLP = spacy.load('en_core_web_lg')
except:
    from subprocess import call
    call('python -m spacy download en_core_web_lg'.split(' '))
    del call
    NLP = spacy.load('en_core_web_lg')

Dataframe = pd.DataFrame
Data = Mapping[int, NamedTuple]
class DependencyParse(NamedTuple):
    token: str
    dep: str
    head: str
    head_pos: str
    children: List[Token]
    doc: Doc
# - 


def nlp(from_data: Union[Dataframe, Data], target_columns: List[str], task: Callable):
    # TODO: gather all functions in here
    raise NotImplementedError('this still needs to be implemented')



def dependency_parse(from_data: Union[Dataframe, Data], target_columns: List[str]) -> Mapping[str, Mapping[int, List[Union[str, Doc]]]]:
    """Dependency parse given documents. Saves a pickled file to `data/feature` folder

    Args:
        from_data: data to parse
        target_columns: specific dataframe columns, or dictionary keys

    Returns:
        dictionary with target_columns mapped towards idx and their dependency parse 
    """
    assert isinstance(target_columns, list)
    res = {k: {} for k in target_columns}
    if isinstance(from_data, Dataframe):
        for i, row in from_data.iterrows():
            for col in target_columns:
                doc = NLP(row[col])
                deps = [DependencyParse(t.text,
                    t.dep_,
                    t.head.text,
                    t.head.pos_,
                    [child for child in t.head.children],
                    doc) 
                    for t in doc]  # extract dependency_parse
                res[col][i] = deps
    to_pickle(res, f'data/features/dependency_parse_{date.today()}.pickle')
    return res


def extract_verbs(from_data: Union[Dataframe, Data], column: str, dependent_on: S) -> Set[str]:
    """Returns the set of lemmatized verbs dependent on given dependency relations.
    See: https://universaldependencies.org/u/dep/

    Args:
        from_data: the data to extract verbs from
        column: column or key to access data
        dependent_on: which syntactical token to depend on

    Returns:
        set of lemmatized verbs
    """
    verbs = set()
    assert isinstance(from_data, Dataframe)
    assert isinstance(column, str)
    data = from_data[column]
    for i in data:
        doc = NLP(i)
        for t in doc:
            if t.dep == dependent_on and t.head.pos == S.VERB:
                verbs.add(t.head.lemma_)
    return verbs


def extract_dependency_parses(from_data: Union[Dataframe, Data], column: str) -> Tuple[Mapping[str, list], Mapping[str, int]]:
    """Gathers docs and maps them to their parse trees and counts them

    Args:
        from_data: the data to extract verbs from
        column: column or key to access data

    Returns:
        dictionary: parsetree to documents
        dictionary: parsetree to count of occurences
    """
    ddict = defaultdict(list)
    cdict = defaultdict(int)
    if NLP is None:
        NLP = spacy.load('en_core_web_lg')
    for i in list(set(from_data[column])):
        doc = NLP(i)
        i = '-'.join([t.dep_ for t in doc])
        ddict[i].append(doc.text)
        cdict[i] += 1

    return (
            {k: v for k, v in sorted_dict(ddict.items())},
            {k: v for k, v in sorted_dict(cdict.items())}
            )


def display_dependency_parse(doc: Doc):
    """Renders dependency parse

    Args:
        doc: spacy doc
    """
    displacy.render(doc)
            

