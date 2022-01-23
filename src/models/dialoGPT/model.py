from sys import argv

from datasets import load_metric
from numpy import argmax
from transformers import AutoModelWithLMHead, PreTrainedModel, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

from src.models.dialoGPT.data import DialoGPTDataset

CHECKPOINT = f'microsoft/dialoGPT-{argv[-1]}'

data = None
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelWithLMHead.from_pretrained(CHECKPOINT)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
args = TrainingArguments(f'dialoGPT-{argv[-1]}-empatheticdialogues-generation',
                         evaluation_strategy='epoch',
                         push_to_hub=True)


def compute_metrics(pred):
    metric = load_metric('perplexity')
    logits, labels = pred
    predictions = argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(model,
                  args,
                  train_dataset=data['train'],
                  eval_dataset=data['validation'],
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics)
