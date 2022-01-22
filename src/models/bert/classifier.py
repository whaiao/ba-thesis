import torch
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

from src.models.bert.data import DailyDialogDataset, TOKENIZER

data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)
args = TrainingArguments('checkpoints/turn_classifier/bert')
model = AutoModelForSequenceClassification.from_pretrained('bert_base_uncased',
                                                           num_labels=5)

train_loader, val_loader, test_loader = DailyDialogDataset.create_dataloaders(
    1)

trainer = Trainer(model,
                  args,
                  train_dataset=train_loader,
                  eval_dataset=val_loader,
                  data_collator=data_collator,
                  tokenizer=TOKENIZER)

trainer.train()
