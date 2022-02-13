from sys import argv
import numpy as np

from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

from src.models.bert.data import create_next_turn_prediction_dataset

checkpoint = argv[-1]
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

data = create_next_turn_prediction_dataset(tokenizer)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
args = TrainingArguments(f'{checkpoint}-dailydialog-turn-classifier',
                         per_device_train_batch_size=1,
                         gradient_accumulation_steps=4,
                         gradient_checkpointing=True,
                         fp16=True)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                           num_labels=4)


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
