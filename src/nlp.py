from collections import defaultdict
from typing import Callable, Tuple, Mapping, NamedTuple, Set, Union, List

from allennlp.predictors.predictor import Predictor
# is needed for the predictor class
import allennlp_models.tagging
import pandas as pd
import spacy
import spacy.symbols as S
from spacy import displacy
from spacy.tokens.doc import Doc
from spacy.tokens.token import Token

from src.constants import DATA_ROOT
from src.utils import sorted_dict
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
# allennlp srl model
SRLPREDICTOR = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
)


class DependencyParse(NamedTuple):
    token: str
    tag: str
    dep: str
    head: str
    head_pos: str
    children: List[Token]


class NounChunk(NamedTuple):
    token: str
    root: str
    root_pos: str
    root_dep: str


class SemanticRoleLabel(NamedTuple):
    parse: str
    verb: str
    tags: List[str]


# -


def nlp(from_data: Union[Dataframe, Data], target_columns: List[str],
        task: Callable):
    # TODO: gather all functions in here
    raise NotImplementedError('this still needs to be implemented')


def dependency_parse(from_sentence: str, return_dep: bool = False) -> dict[Doc, str]:
    """Dependency parse given documents. Saves a pickled file to `data/feature` folder
    See (Spacy Doc)[https://spacy.io/models/en#en_core_web_lg-labels]

    Args:
        from_sentence: sentence to parse

    Returns:
        - doc -> dependency parse structure joined by -
        (opt) list of dependency parse named tuples containing:
            - token text
            - token's POS tag
            - token dependency relation
            - token's head text
            - token's head POS tag
            - token's head children
            - `Spacy` document class
    """
    # IMPORTANT! tokens cannot be pickled, must pickle Doc instead
    sentence = from_sentence
    doc = NLP(sentence)
    parse = {
        doc: '-'.join(t.dep_ for t in doc)
    }

    if return_dep:
        return parse, [DependencyParse(t.text, t.pos_, t.dep_, t.head.text, t.head.pos_, [c for c in t.head.children]) for t in doc]

    return parse


def extract_verbs(from_data: Union[Dataframe, Data], column: str,
                  dependent_on: S) -> Set[str]:
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


def extract_dependency_parses(
        from_data: Union[Dataframe, Data],
        column: str) -> Tuple[Mapping[str, list], Mapping[str, int]]:
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

    return ({k: v
             for k, v in sorted_dict(ddict.items())},
            {k: v
             for k, v in sorted_dict(cdict.items())})


def noun_chunking(from_data: Union[Dataframe, Data],
                  column: str) -> Mapping[str, List[NounChunk]]:
    """Gather noun chunks of documents

    Args:
        from_data: the data to extract verbs from
        column: column or key to access data

    Returns:
        dictionary of document mapped to their noun chunks
    """

    ddict = defaultdict(list)
    for i in list(set(from_data[column])):
        doc = NLP(i)
        ddict[doc.text] = [
            NounChunk(c.text, c.root.head.text, c.root.head.pos_,
                      c.root.head.dep_) for c in doc.noun_chunks
        ]

    return ddict


def display_dependency_parse(doc: Doc):
    """Renders dependency parse

    Args:
        doc: spacy dependency parse to visualize
    """
    displacy.render(doc)


def srl(
    sentence: str,
    predictor: Predictor = SRLPREDICTOR
    ) -> List[SemanticRoleLabel]:
    """Uses AllenNLP semantic role labeling model to tag sentence

    Args:
        text: sentence to tag

    Returns:
        dictionionary with tokenized verb frames tagged with semantic roles
    """

    pred = predictor.predict(sentence=sentence)
    verbs = []
    for verb in pred['verbs']:
        verbs.append(
            SemanticRoleLabel(verb['description'], verb['verb'], verb['tags']))

    return verbs
