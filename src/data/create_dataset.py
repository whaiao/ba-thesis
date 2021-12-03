from typing import List, NamedTuple, Tuple
from multiprocess.pool import AsyncResult

import pandas as pd
import spacy
from spacy.language import Language
from tqdm import tqdm

from preprocessing import social_chem
from ..utils import multiprocess


DataFrame = pd.DataFrame
class Token(NamedTuple):
    lemma: str
    pos: str


def load_atomic(path: str = 'data/processed/v4_atomic_all.tsv') -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', encoding='utf8', index_col='Unnamed: 0')


def create_action_dataset() -> Tuple[pd.DataFrame]:
    """Create dataset from `social chemistry` actions with `atomic` knowledge

    Returns:
        dataframe holding the union of actions and knowledge
    """
    nlp: Language = spacy.load('en_core_web_lg')  # loads large spacy model for tagging and parsing
    sc_data = social_chem()
    sc_actions = sc_data['action']
    atomic_data = load_atomic()
    atomic_actions = atomic_data['prefix']

    def process_actions(doc: str, filter_tag: str = 'VERB') -> List[Token]:
        doc = nlp(doc)
        if doc.has_annotation('TAG'):
            return [Token(token.lemma_, token.pos_) 
                    for token in doc 
                    if token.pos_ == filter_tag]
        else:
            return []

    
    tqdm.pandas()
    sc_actions = sc_actions.apply(lambda x: process_actions(x))
    # lemmatize for verb overlap
    atomic_actions = atomic_actions.apply(lambda x: [t.lemma_ for t in nlp(' '.join(eval(x)))])

    sc_data['extracted_actions'] = sc_actions
    atomic_data['prefix'] = atomic_actions

    print('Saving data')
    atomic_data.to_csv('data/atomic_processed.tsv', sep='\t', encoding='utf8')
    sc_data.to_csv('data/sc_processed.tsv', sep='\t', encoding='utf8')
    print('Saved frames')
    
    return (atomic_data, sc_data)

def retrieve_verb_overlap(d1: DataFrame, d2: DataFrame, multiprocessing: bool = True) -> AsyncResult:
    atomic_cols = [
            'oEffect',
            'oReact',
            'oWant',
            'xAttr',
            'xEffect',
            'xIntent',
            'xNeed',
            'xReact',
            'xWant',
            'prefix'
            ]

    sc_actions = d1['extracted_actions']

    def search(partition: DataFrame, partition_id: int) -> DataFrame:
        """Search for overlap in both dataframes"""
        df = []
        for i, sca in enumerate(sc_actions):
            tmp_dict = {k: [] for k in atomic_data.columns}
            for _, a in partition.iterrows():
                for s in eval(sca):
                    if s.lemma in a['prefix']:
                        for k in tmp_dict.keys():
                            tmp_dict[k].append(a[k])
            tmp_dict['id'] = i  # add id to map back to dataset
            df.append(tmp_dict)

    
        df = pd.DataFrame(df)
        df.to_csv(f'/data/tmp/verb_{partition_id}.tsv', sep='\t', encoding='utf8')  # save intermediate repr
        return df

    if multiprocessing:
        atomic_data = d2[atomic_cols]
        split_val = len(atomic_data) // 4
        splits = [pd.DataFrame(atomic_data.iloc[i-split_val:i]) for i in range(split_val, split_val*5, split_val)]
        iters = iter(zip(splits, range(1,5)))
        res = multiprocess(search, iters)
        return res

