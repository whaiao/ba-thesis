"""Atomic Processing Utils"""

from os import cpu_count
from src.constants import DATA_ROOT, PNAME_PLACEHOLDER_RE, PNAME_SUB
from src.utils import multiprocess_dataset

from functools import partial
from glob import iglob
import pickle
from random import choice
import re
from typing import Dict, List, NamedTuple, Tuple, Callable

from multiprocess import Pool
import pandas as pd
from tqdm import tqdm

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

    def replace(x: str, placeholder: str, replace_with: str) -> str:
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
    print(f'Start {parse_type} parsing')
    parses = []
    for i, t in enumerate(df[col], start=0):
        tmp = fn(t) if isinstance(t, str) else None
        if i % 5000 == 0:
            print('step ', i)
            print('Current sample: ', tmp)
        parses.append(tmp)
    df[f'{col}-{parse_type}'] = parses
    if save:
        df.to_pickle(f'{DATA_ROOT}/atomic/parse-{col}-{parse_type}.pickle')
    return df


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


# Testing area
if __name__ == "__main__":
    from sys import argv
    atomic = load_atomic_data(save=False)
    atomic = fill_placeholders(atomic)
    if argv[-1] == 'mp':
        # we only need the dependency parse
        dp = multiprocess_dataset(parse,
                                  atomic,
                                  save=False,
                                  col='tail',
                                  parse_type='dp')
        with open('./atomic_dp.pickle', 'wb') as f:
            pickle.dump(dp, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        srl = parse(atomic, save=True, col='tail', parse_type='dp')
