import yaml

from pathlib import Path
from typing import Union, Dict

def get_config(yaml_filepath: str) -> Dict[str, Union[str, float, int]]:
    """
    Parses YAML file and returns config dictionary

    Args:
        yaml_filepath: str - filepath

    Returns:
        dict[str, Union[str, float, int]] - training config
    """
    path = Path(yaml_filepath)
    with open(path.resolve(), 'r') as f:
        return yaml.safe_load(f)

