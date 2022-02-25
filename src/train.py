from dataclasses import dataclass
from datetime import datetime
import json
import os
from pprint import pprint
from typing import Callable, Iterable, Iterator, List, Mapping, Tuple, Union

from datasets import load_dataset, load_metric
from datasets.dataset_dict import DatasetDict

from more_itertools import pairwise
import numpy as np

from tqdm.auto import trange, tqdm
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch import nn
from torchmetrics import MetricCollection, BLEUScore
from torchmetrics.text.bert import BERTScore

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import wandb

from src.models.neural_empathy import NeuralEmpathy, ModelConfig
from src.utils import init_from_checkpoint


@dataclass
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 1e-4
    betas: Iterable[float] = None
    gradient_accumulation: bool = True
    report_every: int = 10
    unfreeze_every: int = 2
    warmup_steps: int = 0
    optimizer: Optimizer = AdamW
    scheduler: Callable = get_linear_schedule_with_warmup
    save_to: str = 'checkpoint/model/'


class Trainer:
    def __init__(self, cfg: TrainingConfig, model_cfg: ModelConfig,
                 model: nn.Module, data: DatasetDict):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.data = data
        self.model = model(model_cfg)
        self.optimizer = self.cfg.optimizer(self.model.parameters(),
                                            self.cfg.learning_rate)

        total_training_steps = len(data['train']) * self.cfg.epochs
        self.scheduler = self.cfg.scheduler(
            self.optimizer,
            num_training_steps=total_training_steps,
            num_warmup_steps=self.cfg.warmup_steps)

        self.model_modules = [
            'dialog_transformer', 'dialog_guiding_module', 'lm_head'
        ]

        # datasets

        metrics = MetricCollection([
            #BERTScore(model_name_or_path='distilbert-base-uncased'),
            BLEUScore()
        ])

        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='valid/')

        #wandb.init(entity='benjaminbeilharz', project='ba-thesis')

    @classmethod
    def load_from_config(cls,
                         cfg_path: str,
                         checkpoint_path: str = None,
                         model: nn.Module = None,
                         optimizer: Optimizer = None,
                         data: DatasetDict = None,
                         **kwargs):
        with open(cfg_path, 'r') as json_file:
            cfg = json.load(json_file)

        if checkpoint_path is not None and model is not None and optimizer is not None:
            model, optimizer = init_from_checkpoint(checkpoint_path, model,
                                                    optimizer)
            return cls(cfg=cfg, model=model, data=data, **kwargs)

        else:
            return cls(cfg=cfg, **kwargs)

    def _save_config(self, save_to: str):
        # use vars to convert dataclass to dict
        date = datetime.now()
        date = date.strftime('%d/%m-%H:%M')
        save_to += date
        with open(save_to, 'w') as json_file:
            json.dump(vars(self.cfg), json_file, indent=4)

    def _save_checkpoint(self, save_to: str):
        date = datetime.now()
        date = date.strftime('%d/%m-%H:%M')
        save_to += date

        torch.save(
            {
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict()
            }, save_to)

    def _wandb_log(self, params: Mapping[str, Tensor]):
        pass

    def _predict_and_calculate_metrics(
            self,
            logits: Tensor,
            target: str,
            train: bool = True) -> Tuple[Tensor, Mapping[str, float]]:
        preds = torch.softmax(logits, dim=-1)
        preds = torch.argmax(preds, dim=1).squeeze(0)
        preds = self.model.lm_tokenizer.decode(preds, skip_special_tokens=True)

        if train:
            metrics = self.train_metrics(preds, target)
        else:
            metrics = self.valid_metrics(preds, target)

        return (preds, metrics)

    def _sample(self, sample: Mapping[str, List[str]]) -> Tuple[str, Iterable]:
        sample = sample['conv']
        ctx = sample.pop(0)
        iterable = pairwise(sample)
        return (ctx, iterable)

    def _prepare_dialog_history(self, ctx: str,
                                dialog: Iterable) -> List[List[str]]:
        turns = []
        hist = ctx.replace('_comma_', ',')
        for d in dialog:
            current, next = d
            turns.append([
                hist,
                current.replace('_comma_', ','),
                next.replace('_comma_', ',')
            ])
            hist += f' {current}'

        return turns

    def training_step(self, sample: Iterable[str]):
        self.model.train()
        history, current, next = sample
        self.optimizer.zero_grad()
        out = self.model(history, current, next)
        loss = out.loss
        logits = out.logits

        if not self.cfg.gradient_accumulation:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return logits, loss

    def validation_step(self, sample: Iterable[str]):
        self.model.eval()
        with torch.no_grad():
            history, current, next = sample
            out = self.model(history, current, next)
            loss = out.loss
            logits = out.logits
            return logits, loss

    def run(self):
        os.system('cls')
        for epoch in trange(1, self.cfg.epochs + 1):
            # unfreeze more modules every x epochs
            if epoch % self.cfg.unfreeze_every == 0 and len(
                    self.model_modules) != 0:
                module_name = self.model_modules.pop(0)
                print(f'Unfreezing {module_name} at Epoch: {epoch}')
                self.model._unfreeze_params(module_name)

            best_checkpoint = None
            train_running_loss = []
            for i, sample in enumerate(tqdm(self.data['train']), start=1):
                ctx, dialog = self._sample(sample)
                dialog = self._prepare_dialog_history(ctx, dialog)
                for turn in dialog:
                    logits, loss = self.training_step(turn)
                    loss /= len(dialog)
                    loss.backward()
                    train_running_loss.append(loss.item())
                if self.cfg.gradient_accumulation:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                current_generation, metrics = self._predict_and_calculate_metrics(
                    logits, turn[-1])

                print(
                    f'Epoch: {epoch}\tTraining Loss: {np.mean(train_running_loss)}\tPerplexity: {torch.exp(loss)}\nCurrent Generation: {current_generation}'
                )
                for k, v in metrics.items():
                    print(f'{k}:\t{v}')

            eval_running_loss = 0.0
            for i, sample in enumerate(tqdm(self.data['valid']), start=1):
                ctx, dialog = self._sample(sample)
                dialog = self._prepare_dialog_history(ctx, dialog)
                for turn in dialog:
                    train_running_loss += self.training_step(turn)

            avg_eval_loss = eval_running_loss / len(self.validation_data)
            if best_checkpoint is None or avg_eval_loss < best_checkpoint:
                best_checkpoint = avg_eval_loss
                # add metrics
                print('Saving model checkpoint with metrics:')
                self._save_checkpoint(self.cfg.save_to)


def main():
    cfg = TrainingConfig()
    mcfg = ModelConfig()
    dataset = load_dataset('benjaminbeilharz/empathetic_dialogues_for_lm')

    trainer = Trainer(cfg, mcfg, NeuralEmpathy, dataset)
    trainer.run()


main()
