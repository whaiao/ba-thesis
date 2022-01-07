import pickle
from src.constants import DATA_ROOT
from src.utils import multiprocess_dataset, read_tsv

import pandas as pd

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


def parse(soc_chem: Dataframe,
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
    assert parse_type in ['srl',
                          'dp'], f'Parse type {parse_type} not implemented'
    assert col in soc_chem.columns

    # load and destroy on-demand
    if parse_type == 'srl':
        from src.nlp import srl
        fn = srl
        del srl
    elif parse_type == 'dp':
        from src.nlp import dependency_parse as dp
        fn = dp
        del dp

    df = soc_chem
    print(f'Start {parse_type} parsing')
    parses = []
    for t in df[col]:
        if isinstance(t, str):
            tmp = fn(t) if t != '' else None
        else:
            tmp = None
        parses.append(tmp)
    df[f'{col}-{parse_type}'] = parses
    if save:
        df.to_pickle(
            f'{DATA_ROOT}/social_chemistry/parse-{col}-{parse_type}.pickle')
    return df


def find_relations(soc_chem: Dataframe, column: str) -> Dataframe:
    """Go through social chemistry situation or action parses and extract objects to look up in head

    Args:
        social chemistry - social chemistry dataframe

    Returns:
        dataframe with objects and verbs as columns added
    """
    assert column in soc_chem.columns
    df = soc_chem
    col = column
    obj_extraction = []
    verb_extraction = []

    for sample in df[col]:
        obj_tmp = []
        verb_tmp = []
        # go through tokens in sentence
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


if __name__ == "__main__":
    soc_chem = load_social_chemistry_data(save=False)
    dp_situation = multiprocess_dataset(parse,
                                        soc_chem,
                                        col='situation',
                                        parse_type='dp',
                                        save=False)
    dp_action = multiprocess_dataset(parse,
                                     soc_chem,
                                     col='action',
                                     parse_type='dp',
                                     save=False)
    srl_situation = multiprocess_dataset(parse,
                                         soc_chem,
                                         col='situation',
                                         parse_type='srl',
                                         save=False)
    srl_action = multiprocess_dataset(parse,
                                      soc_chem,
                                      col='action',
                                      parse_type='srl',
                                      save=False)

    filenames = ['dp_situation', 'dp_action', 'srl_situation', 'srl_action']

    for filename, data in zip(
            filenames, [dp_situation, dp_action, srl_situation, srl_action]):
        with open(f'{DATA_ROOT}/social_chemistry/{filename}.pickle',
                  'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
