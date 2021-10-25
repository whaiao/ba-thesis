import wandb
from .dtypes import *


def run_session(experiment_name: str, cfg: Mapping[str, Any], log_dict: Mapping[str, Jndarray], entity: str = 'benjaminbeilharz', model=None):
    wandb.init(project=experiment_name, entity=entity)
    # TODO: check if params are in possible wandb config
    wandb.config = cfg
    wandb.log(log_dict)

    if cfg['session_type'] == 'torch':
        assert model is not None, 'Model cannot be empty if you are using wandb with PyTorch'
        wandb.watch(model)

