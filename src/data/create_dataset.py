from typing import List, NamedTuple
from multiprocess.pool import Pool

import pandas as pd
import spacy
from spacy.language import Language
from tqdm import tqdm

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
        if doc.has_annotation('TAG'):
            return [Token(token.lemma_, token.pos_) 
                    for token in doc 
                    if token.pos_ == filter_tag]
        else:
            return []

    
    tqdm.pandas()
    print('Processing social chem')
    sc_actions = sc_actions.apply(lambda x: process_actions(x))
    # lemmatize for coherent overlap
    print('Processing atomic')
    atomic_actions = atomic_actions.apply(lambda x: [t.lemma_ for t in nlp(' '.join(eval(x)))])

    sc_data['extracted_actions'] = sc_actions
    atomic_data['prefix'] = atomic_actions

    print('Saving data')
    atomic_data.to_csv('data/atomic_processed.tsv', sep='\t', encoding='utf8')
    sc_data.to_csv('data/sc_processed.tsv', sep='\t', encoding='utf8')
    print('Saved frames')
    
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

    # final dataframe
    atomic_data = atomic_data[atomic_cols]
    df = []

    for i, sca in enumerate(sc_actions):
        tmp_dict = {k: [] for k in atomic_data.columns}

        for _, a in atomic_data.iterrows():
            for s in sca:
                if s.lemma in a['prefix']:
                    for k in tmp_dict.keys():
                        tmp_dict[k].append(a[k])
        tmp_dict['id'] = i
        df.append(tmp_dict)

    df = pd.DataFrame(df)
    df.to_csv('data/unified.tsv', sep='\t', encoding='utf8')
    return df

if __name__ == "__main__":
    at = pd.read_csv('data/atomic_processed.tsv', sep='\t', encoding='utf8')
    sc = pd.read_csv('data/sc_processed.tsv', sep='\t', encoding='utf8')
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

    sc_actions = sc['extracted_actions']

# final dataframe
    atomic_data = at[atomic_cols]
    split_val = len(atomic_data) // 4

    splits = [pd.DataFrame(atomic_data.iloc[i-split_val:i]) for i in range(split_val, split_val*5, split_val)]

    def search(at_split):
        df = []
        for i, sca in enumerate(sc_actions):
            tmp_dict = {k: [] for k in atomic_data.columns}
            for _, a in at_split.iterrows():
                for s in eval(sca):
                    if s.lemma in a['prefix']:
                        for k in tmp_dict.keys():
                            tmp_dict[k].append(a[k])
            tmp_dict['id'] = i
            df.append(tmp_dict)

        df = pd.DataFrame(df)
        # df.to_csv(f'data/split{at_split}.tsv', sep='\t', encoding='utf8')
        return df

    with Pool(8) as pool:
        results = pool.map_async(search, splits)
        print(results.get())

