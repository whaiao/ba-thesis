from functools import wraps
from typing import Callable

import wandb



def run_session(session: Session, log_dict: Mapping[str, Jndarray], entity: str = 'benjaminbeilharz', model=None):
    wandb.init(project=session.name, entity=entity)
    # TODO: check if params are in possible wandb config
    wandb.config = session.cfg
    wandb.log(log_dict)

    if session.cfg['session_type'] == 'torch':
        assert model is not None, 'Model cannot be empty if you are using wandb with PyTorch'
        wandb.watch(model)


# def monitor(func: Callable, cfg: dict):
#     @wraps(func)
#     def monitor_experiment(*args, **kwargs):
#         wandb.init(project=cfg['name'], entity='benjaminbeilharz')
#         func(*args, **kwargs)
#         wandb.watch(mod
# 
