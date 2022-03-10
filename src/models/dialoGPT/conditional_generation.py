from functools import partial
from typing import Callable, Dict

from datasets import load_metric
from numpy import argmax

import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorWithPadding

from src.models.bert.data import create_next_turn_prediction_dataset

wandb.init(project='ba-thesis', entity='benjaminbeilharz')

# make sure data is in the right format

EPOCHS = 10.
tokenizer = AutoTokenizer.from_pretrained(f'microsoft/DialoGPT-medium')
model = GPT2LMHeadModel.from_pretrained(f'microsoft/DialoGPT-medium')
data = create_next_turn_prediction_dataset(tokenizer=tokenizer, is_gpt=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenizer.pad_token = tokenizer.eos_token



def compute_metrics(pred):
    metric = load_metric('accuracy', 'perplexity')
    logits, labels = pred
    predictions = argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


args = TrainingArguments('checkpoints/dialoGPT-medium-conditioned-next-turn',
                         num_train_epochs=3.0,
                         save_strategy='epoch',
                         evaluation_strategy='epoch',
                         load_best_model_at_end=True,
                         per_device_train_batch_size=4,
                         per_device_eval_batch_size=1,
                         gradient_accumulation_steps=2,
                         gradient_checkpointing=True,
                         fp16=True,
                         report_to='wandb')
trainer = Trainer(model,
                  args,
                  train_dataset=data['train'],
                  eval_dataset=data['validation'],
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics)

trainer.train()
