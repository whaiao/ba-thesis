#! /usr/bin/env python3
"""
Data reading module
"""

from json_lines import reader
from typing import Any, List, Mapping
import pandas as pd

def read_jsonlines(filepath: str) -> List[Mapping[Any, Any]]:
    """
    Reads `.jsonl` into a list of json-like objects
    """
    with open(filepath, 'rb') as f:
        return [i for i in reader(f)]


def read_csv(filepath: str, seperator: str = ',') -> pd.DataFrame:
    """
    Reads character limited file into dataframe

    Args:
        filepath: Path to file
        seperator: character used to seperate columns
    """
    return pd.read_csv(filepath, encoding='utf8', sep=seperator)
