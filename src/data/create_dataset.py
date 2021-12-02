from typing import List, NamedTuple

import pandas as pd
import spacy
from spacy.language import Language

from preprocessing import social_chem


class Token(NamedTuple):
    lemma: str
    pos: str


def load_atomic(path: str = 'data/processed/v4_atomic_all.tsv') -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', encoding='utf8', index_col='Unnamed: 0')


def create_action_dataset() -> pd.DataFrame:
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
        return [Token(token.lemma_, token.pos_) 
                for token in doc 
                if token.pos_ == filter_tag]

    sc_actions = sc_actions.apply(lambda x: process_actions(x))

    atomic_actions = atomic_actions.apply(lambda x: ' '.join(x))
    atomic_actions = atomic_actions.apply(lambda x: process_actions(x))
    atomic_data['prefix'] = atomic_actions
    
    df = []
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

    atomic_data = atomic_data[atomic_cols]

    for i, sca in enumerate(sc_actions):
        tmp_dict = {k: [] for k in atomic_data.columns}

        for _, a in atomic_data.iterrows():
            for s in sca:
                if s in a['prefix']:
                    for k in tmp_dict.keys():
                        tmp_dict[k].append(a[k])
        tmp_dict['id'] = i
        df.append(tmp_dict)

    df = pd.DataFrame(df)
    df.to_csv('data/unified.tsv', sep='\t', encoding='utf8')
    return df


                    
                    
            
            



    


create_action_dataset()
