from glob import glob
from pprint import pprint
from typing import List, NamedTuple, Tuple

import pandas as pd
import spacy
from multiprocess.pool import AsyncResult
from spacy.language import Language
from tqdm import tqdm

from src.utils import multiprocess
# from src.data.preprocessing import social_chem

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
    sc_data = pd.read_csv('data/processed/social_chem_agg.tsv', sep='\t', encoding='utf8')
    sc_actions = sc_data['action']
    # atomic_data = load_atomic()
    # atomic_actions = atomic_data['prefix']

    def process_actions(doc: str, filter_tag: str = 'VERB') -> List[Token]:
        doc = nlp(doc)
        if doc.has_annotation('TAG'):
            return [Token(token.lemma_, token.pos_) 
                    for token in doc 
                    if token.pos_ == filter_tag]
        else:
            return []

    
    sc_actions = sc_actions.apply(lambda x: process_actions(x))
    # lemmatize for verb overlap
    # atomic_actions = atomic_actions.apply(lambda x: [t.lemma_ for t in nlp(' '.join(eval(x)))])

    sc_data['extracted_actions'] = sc_actions
    # atomic_data['prefix'] = atomic_actions

    print('Saving data')
    # atomic_data.to_csv('data/atomic_processed.tsv', sep='\t', encoding='utf8')
    sc_data.to_csv('data/sc_processed.tsv', sep='\t', encoding='utf8')
    print('Saved frames')
    
    return sc_data #(atomic_data, sc_data)

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


def unify_dataframes(dataframes: List[str]):
    data = []
    for df in tqdm(dataframes):
        data.append(pd.read_csv(df, sep='\t', encoding='utf8', index_col='id'))
    
    return pd.concat(data).reset_index(drop=False).to_csv('./data/tmp/complete_verb_frame.tsv', encoding='utf8', sep='\t')


def merge_verb_data():
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
    social_chem = pd.read_csv('data/sc_processed.tsv', sep='\t', encoding='utf8', index_col='Unnamed: 0')
    verb_frame = pd.read_csv('data/tmp/complete_verb_frame.tsv', sep='\t', encoding='utf8')
    candidate_df = []
    
    for i, _ in social_chem.iterrows():
        tmp = {k: [] for k in atomic_cols}
        for _, a in verb_frame[verb_frame['id'] == i].iterrows():
            for k in tmp.keys():
                tmp[k].extend(eval(a[k]))
        
        candidate_df.append(tmp)

    df = pd.DataFrame(candidate_df)
    df.to_csv('data/tmp/canditate.tsv', sep='\t', encoding='utf8', index=None)
    
    # pd.merge(social_chem, df).to_csv('data/tmp/verbs.tsv', sep='\t', encoding='utf8')


def main():
    atomic_actions = pd.read_csv('data/atomic_processed.tsv', sep='\t', encoding='utf8')
    print('start actions')
    sc_actions = create_action_dataset() 
    print('start overlap')
    retrieve_verb_overlap(sc_actions, atomic_actions)
    files = glob('data/tmp/*split*.tsv')
    print('start unifying')
    unify_dataframes(files)
    print('start merging')
    merge_verb_data()



        
            
    

if __name__ == "__main__":
    main()
