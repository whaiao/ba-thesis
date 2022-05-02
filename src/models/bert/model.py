from sys import argv

import numpy as np

from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

from src_old.models.bert.data import create_datasets

checkpoint = argv[-1]
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

data = create_datasets(tokenizer)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
args = TrainingArguments(f'{checkpoint}-dailydialog-turn-classifier',
                         evaluation_strategy='epoch',
                         push_to_hub=True)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                           num_labels=5)


def compute_metrics(pred):
    metrics = load_metric('accuracy')
    logits, labels = pred
    predictions = np.argmax(logits, dim=-1)
    return metrics.compute(predictions=predictions, references=labels)


trainer = Trainer(model,
                  args,
                  train_dataset=data['train'],
                  eval_dataset=data['validation'],
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics)

trainer.train()
