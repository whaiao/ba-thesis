
from datetime import date
from typing import Iterable, Mapping, NamedTuple, Union, List

import pandas as pd
import spacy
from spacy import displacy
from spacy.tokens.doc import Doc

from src.utils import read_tsv
from src.data.save import to_pickle

NLP = spacy.load('en_core_web_lg')
Dataframe = pd.DataFrame
Data = Mapping[int, NamedTuple]

class DependencyParse(NamedTuple):
    token: str
    dep: str
    doc: Doc


def dependency_parse(data: Union[Dataframe, Data], target_columns: List[str]) -> Mapping[str, Mapping[int, List[Union[str, Doc]]]]:
    assert isinstance(target_columns, Iterable)
    res = {k: {} for k in target_columns}
    if isinstance(data, Dataframe):
        for i, row in data.iterrows():
            for col in target_columns:
                doc = NLP(row[col])
                deps = [DependencyParse(t, t.dep_, doc) for t in doc]  # extract dependency_parse
                res[col][i] = deps
    to_pickle(res, f'data/features/dependency_parse_{date.today()}')
    return res


if __name__ == "__main__":
    df = read_tsv('data/processed/social_chem_agg.tsv')
    res = dependency_parse(df, ['action'])
    __import__('pprint').pprint(res)




