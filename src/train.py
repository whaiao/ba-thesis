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
from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import wandb

from src.models.neural_empathy import NeuralEmpathy, ModelConfig
from src.utils import init_from_checkpoint


@dataclass
class TrainingConfig:
    """Training Configuration for `Manager` class"""
    epochs: int = 6
    learning_rate: float = 1e-3
    betas: Iterable[float] = None
    gradient_accumulation: bool = True
    report_every: int = 10
    unfreezing_modules: List[str] = field(default_factory=lambda: ['dialog_guiding_module'])
    unfreeze_every: int = 1
    warmup_steps: int = 0
    scheduler: Callable = get_linear_schedule_with_warmup
    save_to: str = 'checkpoints/models/tdec_fixed'

@dataclass
class GenerationConfig:
    """Generation Configuration for LM Generation"""
    num_beams: int = 10
    max_length: int = 200
    no_repeat_ngram_size: int = 2
    num_return_sequences: int = 5
    early_stopping: bool = True



class Manager:
    """Manager class handles training, validation and testing"""
    def __init__(self, cfg: TrainingConfig, model_cfg: ModelConfig,
            model: nn.Module, optimizer: Optimizer,
            data: DatasetDict, _pretrained: bool = False):
        """Initializes Manager

        Args:
            cfg - training configuration file containing hyperparams for experiments
            model_cfg - model configuration file containing hyperparams for model
            model - neural network
            optimizer - `torch` or `transformers` optimizer class
            data - dataset pulled from `HuggingFace`
            _pretrained - sets models and optimizers based on if initialized from checkpoint
        """
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.cfg = cfg
        self.model_cfg = model_cfg
        if _pretrained:
            self.model = model
            self.optimizer = optimizer
        else:
            self.model = model(model_cfg).to(self.device)
            if optimizer.__name__ == 'Adafactor':
                self.optimizer = optimizer(self.model.parameters(),
                            lr=1e-3,
                            eps=(1e-30, 1e-3),
                            clip_threshold=1.0,
                            decay_rate=-0.8,
                            beta1=None,
                            weight_decay=0.0,
                            relative_step=False,
                            scale_parameter=False,
                            warmup_init=False,
                            )
            else:
                self.optimizer = optimizer(self.model.parameters(),
                        lr=self.cfg.learning_rate)

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
                         data: DatasetDict = None):
        """Creates Manager class based on existing checkpoints and configurations

        Args:
            cfg_path - path to pickled configuration file
            model_cfg_path - path to pickled model configuration file
            checkpoint_path - path to model and optimizer checkpoint path
            model - model to load
            optimizer - optimizer to load
            data - dataset from `HuggingFace`
        """
        with open(cfg_path, 'rb') as p:
            cfg = pickle.load(p)

        with open(model_cfg_path, 'rb') as p:
            mcfg = pickle.load(p)

        model = model(cfg)
        if optimizer.__name__ == 'Adafactor':
            optimizer = optimizer(model.parameters(),
                        lr=1e-3,
                        eps=(1e-30, 1e-3),
                        clip_threshold=1.0,
                        decay_rate=-0.8,
                        beta1=None,
                        weight_decay=0.0,
                        relative_step=False,
                        scale_parameter=False,
                        warmup_init=False,
                        )
        else:
            optimizer = optimizer(model.parameters(), lr=cfg.learning_rate)

        # loading model checkpoint
        model, optimizer = init_from_checkpoint(checkpoint_path, model, optimizer)

        print('Successfully loaded config file')
        return cls(cfg=cfg, model_cfg=mcfg, model=model, optimizer=optimizer, data=data, _pretrained=True)


    def _save_config(self, save_to: str):
        """Saves current config with a timestamp
        
        Args:
            save_to - path to save config to
        """
        date = datetime.now()
        date = date.strftime('%d-%m-%H')
        save_to += f'train_cfg_{date}'
        with open(save_to, 'wb+') as p:
            pickle.dump(self.cfg, p)
        with open(save_to.replace('train', 'model'), 'wb+') as p:
            pickle.dump(self.model_cfg, p)

    def _save_checkpoint(self, save_to: str):
        """Saves current checkpoint /w timestamp

        Args:
            save_to - path to save checkpoint to
        """
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
        """Makes a prediction and calculates metrics

        Args:
            logits - current prediction from `lm_head`
            target - gold response
            train - true by default, whether to output validation or train metrics

        Returns:
            generated output tensor and a metric dictionary
        """
        target = target['next']
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
        """Creates a sample from the dataset

        Args:
            sample - an entry from dataset

        Returns:
            context and complete dialog
        """
        sample = sample['conv']
        ctx = sample.pop(0)
        iterable = pairwise(sample)
        return (ctx, iterable)

    def _prepare_dialog_history(self, ctx: str,
                                dialog: Iterable) -> List[List[str]]:
        """Prepares the dialog in pairs for the language model task
        
        Args:
            ctx - context of dialog
            dialog - iterable of utterances

        Returns:
            list of history, current and next turn
        """
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

    def training_step(self, sample: Iterable[str]) -> Tuple[Tensor]:
        """Training step

        Args:
            sample - dialog history, current utterance and gold response

        Returns:
            tuple of logits and loss
        """
        self.model.train()
        history = sample['history']
        current = sample['current']
        nxt = sample['next']
        self.model.zero_grad()
        self.optimizer.zero_grad()
        out = self.model(history, current, nxt)
        loss = out.loss
        logits = out.logits
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return logits, loss

    def validation_step(self, sample: Iterable[str]):
        """Validation step

        Args:
            sample - dialog history, current utterance and gold response

        Returns:
            tuple of logits and loss
        """
        self.model.eval()
        with torch.no_grad():
            history = sample['history']
            current = sample['current']
            nxt = sample['next']
            out = self.model(history, current, nxt)
            loss = out.loss
            logits = out.logits
            return logits, loss


    def inference_step(self, dialog_history: str, current_utterance: str, **generation_settings) -> str:
        """Inference

        Args:
            dialog_history - the dialog history as a concatenated string
            current_utterance - the current utterance
            **generation_settings - settings for generation

        Returns:
            generated response
        """
        self.model.eval()
        with torch.no_grad():
            if generation_settings is not None:
                out = self.model.inference(dialog_history, current_utterance, **generation_settings)
            else:
                out = self.model.inference(dialog_history, current_utterance)
        return out


    def run(self):
        """Runs a complete cycle of epochs, training and validation"""
        for epoch in trange(1, self.cfg.epochs + 1):
            best_checkpoint = None
            train_running_loss = []
            for i, sample in enumerate(tqdm(self.data['train']), start=1):
                logits, loss = self.training_step(sample)
                perplexity = torch.exp(loss)
                log_key = 'train/loss'
                wandb.log({log_key: loss.item(),
                    'train/perplexity': perplexity,
                    'train/epoch_progress': i/len(self.data['train'])})
                train_running_loss.append(loss.item())

                current_generation, metrics = self._predict_and_calculate_metrics(
                    logits, sample)

                print(
                        f'Epoch: {epoch}\tTraining Loss: {np.mean(train_running_loss)}\tPerplexity: {perplexity}\nCurrent Generation: {current_generation}'
                )
                if i % 10000 == 0:
                    self._save_config(self.cfg.save_to)
                    self._save_checkpoint(self.cfg.save_to)

            print(
                f'Finished Epoch: {epoch}\t Average Training Loss: {np.mean(train_running_loss)}'
            )
            print(f'Saving Model with Average Training Loss {np.mean(train_running_loss)}')

            eval_running_loss = []
            for i, sample in enumerate(tqdm(self.data['validation']), start=1):
                logits, validation_loss = self.validation_step(sample)
                perplexity = torch.exp(validation_loss)
                wandb.log({'validation/loss': validation_loss.item(),
                    'validation/perplexity': perplexity,
                    'validation/epoch_progess': i/len(self.data['validation'])})
                eval_running_loss.append(validation_loss.item())

                current_val_generation, val_metrics = self._predict_and_calculate_metrics(logits, sample, train=False)
            print(
                    f'Epoch: {epoch}\tValidation Loss: {np.mean(eval_running_loss)}\tValidation Perplexity: {torch.exp(validation_loss)}\nCurrent Generation: {current_val_generation}'
                    )

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
    gcfg = GenerationConfig().__dict__
    #print('Config: ', gcfg)
    dataset = load_dataset('benjaminbeilharz/ed-for-lm')
    #mcfg.lm_checkpoint = 'benjaminbeilharz/t5-conditioned-next-turn'

    # trainer = Manager(cfg, mcfg, NeuralEmpathy, Adafactor, dataset)
    #trainer = Manager(cfg, mcfg, NeuralEmpathy, AdamW, dataset)
    checkpoint_path = 'checkpoints/models/'
    # bert knowledge encoder
    #checkpoint = checkpoint_path + 'adamw-enc-dec-bert-knowledgeencodercheckpoint_20-03-17.pt'
    # model_cfg = checkpoint_path + 'adamw-enc-dec-bert-knowledgeencodermodel_cfg_20-03-17'
    # train_cfg = checkpoint_path + 'adamw-enc-dec-bert-knowledgeencodertrain_cfg_20-03-17'
    # transformer knowledge encoder
    checkpoint = checkpoint_path + 'neural_empath_with_enc_deccheckpoint_25-03-09.pt'
    model_cfg = checkpoint_path + 'neural_empath_with_enc_decmodel_cfg_25-03-09'
    train_cfg = checkpoint_path + 'neural_empath_with_enc_dectrain_cfg_25-03-09'
    # checkpoint = checkpoint_path + 'adafactor-enc-deccheckpoint_24-03-09.pt'
    # model_cfg = checkpoint_path + 'adafactor-enc-decmodel_cfg_24-03-09'
    # train_cfg = checkpoint_path + 'adafactor-enc-dectrain_cfg_24-03-09'
    # trainer = Manager.load_from_config(train_cfg, model_cfg, checkpoint, NeuralEmpathy, Adafactor, dataset)

    # EVAL:
    test_data = load_dataset('benjaminbeilharz/ed-for-lm', split='test')
    trainer = Manager.load_from_config(train_cfg, model_cfg, checkpoint, NeuralEmpathy, AdamW, dataset)
    for sample in test_data:
        history, current, _ = sample
        trainer.inference_step(dialog_history=history, current_utterance=current, **gcfg)

main()
