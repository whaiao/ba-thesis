import functools

import jax
import numpy as np
import optax
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from transformers import PreTrainedModel, DataCollatorWithPadding

from .dtypes import *

DataCollator = DataCollatorWithPadding
LossFn = nn.Module
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')


class JAXTrainer:
    def __init__(self, net_init, loss_fn,
                 optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, master_rng, data: Data):
        out_rng, init_rng = jax.random.split(master_rng, 2)
        params = self._net_init(init_rng, data)
        opt_state = self._opt.init(params)
        out = {
            step: np.array(0),
            rng: out_rng,
            opt_state: opt_state,
            params: params
        }
        return out

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: Mapping[str, Any], data: Data):
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        loss, grads = jax.value_and_grad(self._loss_fn)(params, rng, data)
        updates, opt_state = self._opt.update(g, state['opt_state'])
        params = optax.apply_updates(params, updates)

        new_state = {
            step: state['step'] + 1,
            rng: new_rng,
            opt_state: opt_state,
            params: params,
        }

        metrics = {
            step: state['step'],
            loss: loss,
        }

        return new_state, metrics


class TorchTrainer:
    def __init__(self,
                 model: PreTrainedModel,
                 optimizer: optim.Optimizer,
                 scheduler,
                 criterion: LossFn,
                 collator: DataCollator,
                 train_data: DataLoader,
                 val_data: DataLoader,
                 test_data: DataLoader,
                 objective: str,
                 accelerate: bool = False):
        self.model = model
        self.optim = optimizer(model.parameters())
        self.scheduler = scheduler
        self.criterion = criterion
        self.collator = collator
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.objective = objective

        self.accelerator = None if accelerate == False else Accelerator()

    def _init_accelerator(self, datasplit: DataLoader):
        self.model, self.optimizer, datasplit = self.accelerator.prepare(
            self.model, self.optim, datasplit)

    def train(self):
        self.model.train()
        for i, batch in enumerate(self.train_data):
            loss = self.step(batch)

    def step(self, batch):
        if self.objective == 'classifications':
            x, Y = batch.to(device)
            self.model.zero_grad()
            out = self.model(**x, labels=Y)
            loss = out.loss
            if self.accelerator is None:
                loss.backward()
            else:
                self.accelerator.backward(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss

    def eval(self):
        self.model.eval()
        pass
