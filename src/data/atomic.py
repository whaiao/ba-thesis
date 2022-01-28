"""Atomic Processing Utils"""

from src.constants import DATA_ROOT, PNAME_PLACEHOLDER_RE, PNAME_SUB

from collections import defaultdict, Counter
from functools import partial
from glob import iglob
import pickle
from random import choice
import re
from typing import Dict, List, NamedTuple, Tuple

import pandas as pd
from tqdm import tqdm
from spacy.tokens.doc import Doc
import spacy.symbols as S

read_tsv = partial(pd.read_csv, sep='\t', encoding='utf8', header=None)
Dataframe = pd.DataFrame


class Relation(NamedTuple):
    relation: str
    tail: str


def load_atomic_data(glob_path: str = f'{DATA_ROOT}/atomic/*.tsv',
                     save: bool = False) -> Dataframe:
    """Load atomic dataset from glob path

    Args:
        glob_path - path to atomic folder
        save - if true saves dataframe to disk

    Returns:
        atomic dataframe
    """

    data_dict = {'head': [], 'relation': [], 'tail': []}

    for file in iglob(glob_path, recursive=False):
        if file.endswith('processed.tsv'): continue
        print(file)
        tmp = read_tsv(file, low_memory=False)
        for i, k in enumerate(data_dict.keys()):
            # retrieve columns and put them into data_dict
            data_dict[k].extend(tmp.iloc[:, i].tolist())

    df = Dataframe.from_dict(data_dict).reset_index().set_index('index')
    if save:
        df_path = f'{DATA_ROOT}/atomic/processed.tsv'
        serialized = f'{DATA_ROOT}/atomic/atomic.pickle'
        print('Saving dataframe to ', df_path)
        df.to_pickle(serialized)
        df.to_csv(df_path, sep='\t', encoding='utf8')
        print('data saved')

    return df


def physical_entity_attributes(
        atomic: Dataframe) -> Tuple[Dataframe, Dataframe]:
    """Extracts physical and entity attributes from Atomic
    
    Args:
        atomic - atomic dataframe

    Returns:
        dataframe only containing physical-entity attributes
    """
    phy_attrs = ['ObjectUse', 'AtLocation', 'MadeUpOf', 'HasProperty']
    ent_attrs = ['CapableOf', 'Desires', 'NotDesires']

    phy_frame = atomic[~atomic['relation'].isin(phy_attrs)]
    ent_frame = atomic[~atomic['relation'].isin(ent_attrs)]

    return (phy_frame, ent_frame)


def social_attributes(atomic: Dataframe) -> Dataframe:
    """Extracts social attributes from Atomic
    
    Args:
        atomic - atomic dataframe

    Returns:
        dataframe only containing social attributes
    """
    attrs = [
        'xNeed', 'xAttr', 'xEffect', 'xReact', 'xWant', 'xIntent', 'oEffect',
        'oReact', 'oWant'
    ]

    df = atomic[~atomic['relation'].isin(attrs)]

    return df


def event_attributes(atomic: Dataframe) -> Tuple[Dataframe, Dataframe]:
    """Extracts event attributes from Atomic
    
    Args:
        atomic - atomic dataframe

    Returns:
        dataframe only containing event attributes
    """
    script_attrs = ['isAfter', 'isBefore', 'HasSubevent']
    dynamic_attrs = ['Causes', 'HinderedBy', 'xReason']
    script_frame = atomic[~atomic['relation'].isin(script_attrs)]
    dyna_frame = atomic[~atomic['relation'].isin(dynamic_attrs)]

    return (script_frame, dyna_frame)


def collect_sample(from_head: str,
                   atomic: Dataframe) -> Dict[str, List[Relation]]:
    """Searches atomic dataframe for head

    Args:
        from_head - string to query
        atomic - atomic dataframe

    Returns:
        dict of head with entries of head
    """
    df = atomic[atomic['head'] == from_head]

    return {
        from_head: [
            Relation(r, t)
            for r, t in zip(df['relation'].tolist(), df['tail'].tolist())
        ]
    }


def fill_placeholders(atomic: Dataframe,
                      columns: List[str] = ['head', 'tail']) -> Dataframe:
    """Fills placeholder values Person {X, Y, Z} with an arbitrary `pname`

    Args:
        atomic - atomic dataframe
        columns - columns to process

    Returns:
        dataframe with replaced names
    """
    df = atomic
    names = PNAME_SUB
    df['origin'] = df['head']

    def fill_obj(row):
        """Fills ___ placeholder"""
        if row['relation'] == 'isFilledBy':
            return row['head'].replace('___', row['tail'])
        else:
            return row['head']

    def replace(x: str, placeholder: str, replace_with: str) -> str:
        """Replaces Person ϵ [X-Z] with natural names"""
        if isinstance(x, str) and re.search(placeholder, x) is not None:
            s = re.sub(placeholder, replace_with, x)
            return s
        return x

    for i in PNAME_PLACEHOLDER_RE:
        rex = re.compile(i, flags=re.IGNORECASE)
        name = choice(names)

        for col in columns:
            df[col] = df[col].apply(
                lambda x: replace(x, placeholder=rex, replace_with=name))

        # do not allow duplicate names
        names.remove(name)
        re.purge()

    for i, row in df.iterrows():
        df.loc[i, 'head'] = fill_obj(row)

    # spacy parses recognizes the underscore as a dobj
    # for coverage i sub the triple underscores with a single one
    # so if we want to retrieve atomic values with the dp structure
    # we can collect more candidates for BERT score.
    df['head'] = df['head'].apply(lambda x: x.replace('___', '_'))

    # TODO: can be done after parsing
    # filter remaining ___ placeholder values (they are not important for our dependency_parses
    # df = df[~df['head'].str.contains('___')]

    return df


def parse(atomic: Dataframe,
          col: str,
          parse_type: str,
          save: bool = True) -> Dataframe:
    """Apply parse function on atomic heads

    Args:
        atomic - atomic dataframe
        parse_type - possible parse types: srl, dp
        save - if true saves dataframe to disk

    Returns:
        dataframe with added parse column
    """
    assert parse_type in ['srl', 'dp'], 'Parse type supplied not implemented'
    assert col in atomic.columns

    if parse_type == 'srl':
        from src.nlp import srl
        fn = srl
        del srl
    elif parse_type == 'dp':
        from src.nlp import dependency_parse as dp
        fn = dp
        del dp

    df = atomic
    print(f'Start {parse_type} parsing, with {len(df[col].unique())} samples.')
    parses = {}

    for i, t in enumerate(tqdm(df[col].unique()), start=0):
        if isinstance(t, str):
            origin = df.loc[df[col] == t]['origin'].iloc[0]
            # pass origin only for dep parse
            tmp = fn([t, origin])
        # tmp = fn(t) if isinstance(t, str) else None
        # dict -> Doc/str: parse
        if i % 5000 == 0:
            print('Current sample: ', tmp)
        parses.update(tmp)

    if save:
        with open(f'{DATA_ROOT}/atomic/parse-{parse_type}.pickle', 'wb') as p:
            pickle.dump(parses, p)
        # df.to_pickle(f'{DATA_ROOT}/atomic/parse-{parse_type}.pickle')
    return df


def create_lookup_dict(dp_parse: str = 'data/atomic/dp.pickle'):
    with open(dp_parse, 'rb') as p:
        parses = pickle.load(p)

    # keys are spacy docs
    ndict = {}
    for p, d in parses.items():
        tmp_verb = []
        tmp_objs = []
        atomic_string = d['origin']
        for t in p:
            if t.pos == S.VERB:
                tmp_verb.append(t.lemma_)
            if 'obj' in t.dep_:
                tmp_objs.append(t.text)
        ndict[p] = {
            'text': atomic_string,
            'verbs': tmp_verb,
            'objects': tmp_objs
        }

    with open(f'data/atomic/lookup.pickle', 'wb') as p:
        pickle.dump(ndict, p)


def count_dict(
        from_pickle: str) -> tuple[Dict[str, int], Dict[str, List[Doc]]]:
    """Returns a count dictionary and a reduced dict to parse structure

    Args:
        from_pickle: path to pickle file

    Returns:
        - dependency parse structure -> count
        - dependency parse structure -> list of collected spacy documents"""

    with open(from_pickle, 'rb') as p:
        # doc -> dep_parse
        data = pickle.load(p)

    counter = Counter([parse for parse in data.values()])
    gatherer = defaultdict(list)
    for k, v in data.items():
        gatherer[v].append(k)

    return (counter, gatherer)


def find_relations(atomic: Dataframe) -> Dataframe:
    """Go through atomic tail parses and extract objects to look up in head

    Args:
        atomic - atomic dataframe

    Returns:
        dataframe with objects and verbs as columns added
    """
    assert 'tail-dp' in atomic.columns
    df = atomic

    obj_extraction = []
    verb_extraction = []
    for sample in df['tail-dp']:
        obj_tmp = []
        verb_tmp = []
        for parse in sample:
            if 'dobj' in parse.dep or 'pobj' in parse.dep:
                obj_tmp.append(parse.text)
                print(f'Sample found in {sample}: {parse.text}')
            else:
                obj_tmp.append(None)

            if 'verb' in parse.tag:
                verb_tmp.append(None)
            else:
                verb_tmp.append(None)

        obj_extraction.append(obj_tmp)
        verb_extraction.append(verb_tmp)

    df['objects'] = obj_extraction
    df['verbs'] = verb_extraction

    return df
