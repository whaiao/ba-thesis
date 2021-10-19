#! /usr/bin/env python3

"""
Configuration handling for experiments
"""
import yaml
from pathlib import Path
from typing import Union, Mapping

def get_config(yaml_filepath: str) -> Mapping[str, Union[str, float, int]]:
    """
    Parses YAML file and returns config dictionary

    Args:
        yaml_filepath: file path to yaml config

    Returns:
        A dictionary of experiment configuration settings
    """
    path = Path(yaml_filepath)
    with open(path.resolve(), 'r') as f:
        return yaml.safe_load(f)

