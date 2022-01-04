from src.constants import DATA_ROOT
# from src.nlp import srl, dependency_parse

from functools import partial
from glob import iglob

import pandas as pd

read_tsv = partial(pd.read_csv, sep='\t', encoding='utf8')
Dataframe = pd.DataFrame


def load_social_chemistry_data(
        path: str = f'{DATA_ROOT}/social_chemistry/social-chem-101.v1.0.tsv',
        save: bool = False) -> Dataframe:
    """Load social chemistry dataset from glob path

    Args:
        path - path to social chemistry file
        save - if true saves dataframe to disk

    Returns:
        social chemistry dataframe
    """
    df = read_tsv(path).convert_dtypes()

    if save:
        df_path = f'{DATA_ROOT}/social_chemistry/processed.tsv'
        serialized = f'{DATA_ROOT}/social_chemistry/social_chemistry.pickle'
        print('Saving dataframe to ', df_path)
        df.to_pickle(serialized)
        df.to_csv(df_path, sep='\t', encoding='utf8')
        print('data saved')

    return df


def extract_rot(from_action: str):
    """Extract rule-of-thumb attributes from social_chemistry
    
    Args:
        from_action - string containing an action
    """
    attrs = ['rot', 'rot-agree', 'rot-categorization']
    pass


if __name__ == "__main__":
    data = load_social_chemistry_data(save=False)
