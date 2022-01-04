"""Atomic Processing Utils"""

from src.constants import DATA_ROOT

from functools import partial
from glob import iglob
from typing import Dict, List, NamedTuple, Tuple

import pandas as pd

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


def connect_entries(atomic: Dataframe) -> Dataframe:
    """Collects all rows with same event and saves it into a new dataframe

    Args:
        atomic: dataframe

    Returns:
        new organized dataframe
    """
    raise NotImplementedError()


if __name__ == "__main__":
    atomic = load_atomic_data(save=False)
