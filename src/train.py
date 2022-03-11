from dataclasses import dataclass, field
from datetime import datetime
import pickle
from pprint import pprint
from re import sub
from typing import Callable, Iterable, Iterator, List, Mapping, Tuple, Union

from accelerate import Accelerator
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
    epochs: int = 6
    learning_rate: float = 1e-4
    betas: Iterable[float] = None
    gradient_accumulation: bool = True
    report_every: int = 10
    unfreezing_modules: List[str] = field(default_factory=lambda: ['dialog_guiding_module'])
    unfreeze_every: int = 1
    warmup_steps: int = 100
    scheduler: Callable = get_linear_schedule_with_warmup
    save_to: str = 'checkpoints/models/'


class Trainer:
    def __init__(self, cfg: TrainingConfig, model_cfg: ModelConfig,
            model: nn.Module, optimizer: Optimizer,
            data: DatasetDict):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.cfg = cfg
        self.model_cfg = model_cfg
        # TODO: init model in load_from_checkpoint
        self.model = model
        self.optimizer = optimizer
        # self.model = model(model_cfg).to(self.device)
        # self.optimizer = optimizer(self.model.parameters(),
        #         self.cfg.learning_rate)
        self.model.cuda()
        self.data = data

        self.model, self.optimizer, self.data['train'] = self.accelerator.prepare(self.model, self.optimizer, self.data['train'])

        total_training_steps = len(data['train']) * self.cfg.epochs
        self.scheduler = self.cfg.scheduler(
            self.optimizer,
            num_training_steps=total_training_steps,
            num_warmup_steps=self.cfg.warmup_steps)

        self.freeze_model_modules = ['dialog_transformer', 'lm_head']
        self.unfreeze_model_modules = self.cfg.unfreezing_modules

        # for module in self.freeze_model_modules:
        #     for p in getattr(self.model, module).parameters():
        #         p.requires_grad = False

        # datasets

        metrics = MetricCollection([
            BLEUScore(2),
        ])

        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='validation/')

        config_dict = self.cfg.__dict__.update(self.model_cfg.__dict__)
        pprint(config_dict)

        wandb.init(entity='benjaminbeilharz', project='ba-thesis', config=config_dict)
        wandb.watch(self.model)

    @classmethod
    def load_from_config(cls,
                         cfg_path: str,
                         model_cfg_path: str,
                         checkpoint_path: str = None,
                         model: nn.Module = None,
                         optimizer: Optimizer = None,
                         data: DatasetDict = None,
                         **kwargs):

        with open(cfg_path, 'rb') as p:
            cfg = pickle.load(p)

        with open(model_cfg_path, 'rb') as p:
            mcfg = pickle.load(p)

        model = model(cfg)
        optimizer = optimizer(model.parameters(), cfg.learning_rate)

        # loading model checkpoint
        model, optimizer = init_from_checkpoint(checkpoint_path, model, optimizer)

        print('Successfully loaded config file')
        return cls(cfg=cfg, model_cfg=mcfg, model=model, optimizer=optimizer, data=data)


    def _save_config(self, save_to: str):
        # use vars to convert dataclass to dict
        date = datetime.now()
        date = date.strftime('%d-%m-%H')
        save_to += f'train_cfg_{date}'
        with open(save_to, 'wb+') as p:
            pickle.dump(self.cfg, p)
        with open(save_to.replace('train', 'model'), 'wb+') as p:
            pickle.dump(self.model_cfg, p)

    def _save_checkpoint(self, save_to: str):
        date = datetime.now()
        date = date.strftime('%d-%m-%H')
        save_to += f'checkpoint_{date}.pt'
        torch.save(
            {
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict()
            }, save_to)

    def _predict_and_calculate_metrics(
            self,
            logits: Tensor,
            target: str,
            train: bool = True) -> Tuple[Tensor, Mapping[str, float]]:
        preds = torch.argmax(logits, dim=-1)
        generated = self.model.lm_tokenizer.decode(preds[0], skip_special_tokens=True)
        postprocessed = sub('\s+', ' ', generated)

        if train:
            metrics = self.train_metrics([postprocessed], [[target]])
        else:
            metrics = self.valid_metrics([postprocessed], [[target]])

        wandb.log(metrics)

        return (postprocessed, metrics)

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


    def inference_step(self, dialog_history: str, current_utterance: str, **generation_settings):
        self.model.eval()
        with torch.no_grad():
            if generation_settings is not None:
                out = self.model.inference(dialog_history, current_utterance, **generation_settings)
            else:
                out = self.model.inference(dialog_history, current_utterance)
        return out


    def run(self):
        for epoch in trange(1, self.cfg.epochs + 1):
            # unfreeze more modules every x epochs
            # if epoch - 1 % self.cfg.unfreeze_every == 0 and len(
            #         self.unfreeze_model_modules) != 0:
            #     module_name = self.unfreeze_model_modules.pop(0)
            #     print(f'Unfreezing {module_name} at Epoch: {epoch}')
            #     self.model._unfreeze_params(module_name)

                

            best_checkpoint = None
            train_running_loss = []
            #quarter_epoch = len(self.data['train']) // 4
            #half_epoch = len(self.data['train']) // 2
            for i, sample in enumerate(tqdm(self.data['train']), start=1):
                #if i == quarter_epoch:
                #    print('Unfreezed Language Model head')
                #    self.model._unfreeze_params(self.freeze_model_modules[-1])

                #if i == quarter_epoch and epoch == 2:
                #    print('Unfreezing Dialog History Encoder')
                #    self.model._unfreeze_params(self.freeze_model_modules[0])

                ctx, dialog = self._sample(sample)
                dialog = self._prepare_dialog_history(ctx, dialog)
                for turn in dialog:
                    logits, loss = self.training_step(turn)
                    perplexity = torch.exp(loss)
                    #log_key = 'train/loss' if i <= quarter_epoch else 'train/lm/loss'
                    log_key = 'train/loss'
                    wandb.log({log_key: loss.item(),
                        'train/perplexity': perplexity})
                    # normalize loss across number of dialog turns
                    train_running_loss.append(loss.item())
                    loss /= len(dialog)
                    loss.backward()
                if self.cfg.gradient_accumulation:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()


                current_generation, metrics = self._predict_and_calculate_metrics(
                    logits, turn[-1])

                print(
                        f'Epoch: {epoch}\tTraining Loss: {np.mean(train_running_loss)}\tPerplexity: {perplexity}\nCurrent Generation: {current_generation}'
                )
                for k, v in metrics.items():
                    print(f'{k}:\t{v}')

            print(
                f'Finished Epoch: {epoch}\t Average Training Loss: {np.mean(train_running_loss)}'
            )
            print(f'Saving Model with Average Training Loss {np.mean(train_running_loss)}')
            self._save_config(self.cfg.save_to)
            self._save_checkpoint(self.cfg.save_to)

            eval_running_loss = []
            for i, sample in enumerate(tqdm(self.data['validation']), start=1):
                ctx, dialog = self._sample(sample)
                dialog = self._prepare_dialog_history(ctx, dialog)
                for turn in dialog:
                    logits, validation_loss = self.validation_step(turn)
                    perplexity = torch.exp(validation_loss)
                    wandb.log({'validation/loss': validation_loss.item(),
                        'validation/perplexity': perplexity})
                    eval_running_loss.append(validation_loss.item())

                current_val_generation, val_metrics = self._predict_and_calculate_metrics(logits, turn[-1], False)
                print(
                        f'Epoch: {epoch}\tValidation Loss: {np.mean(eval_running_loss)}\tValidation Perplexity: {torch.exp(validation_loss)}\nCurrent Generation: {current_val_generation}'
                        )
                for k, v in val_metrics.items():
                    print(f'{k}:\t{v}')
                

            avg_eval_loss = np.mean(eval_running_loss)
            if best_checkpoint is None or avg_eval_loss < best_checkpoint:
                best_checkpoint = avg_eval_loss
                # add metrics
                print('Saving model checkpoint with metrics:')
                self._save_config(self.cfg.save_to)
                for k,v in val_metrics.items():
                    print(f'{k}:\t{v}')
                    print(f'Avg Eval Loss:\t{avg_eval_loss}')
                self._save_checkpoint(self.cfg.save_to)


def main():
    cfg = TrainingConfig()
    mcfg = ModelConfig()
    dataset = load_dataset('benjaminbeilharz/empathetic_dialogues_for_lm')

    #trainer = Trainer(cfg, mcfg, NeuralEmpathy, AdamW, dataset)
    checkpoint_path = 'checkpoints/models/'
    trainer_cfg = checkpoint_path + 'train_cfg_10-03-23'
    model_cfg = checkpoint_path + 'model_cfg_10-03-23'
    checkpoint = checkpoint_path + 'checkpoint_10-03-23.pt'
    trainer = Trainer.load_from_config(trainer_cfg, model_cfg, checkpoint, NeuralEmpathy, AdamW, dataset)
    trainer.run()


main()
