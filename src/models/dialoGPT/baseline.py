from pprint import pprint

from datasets import load_dataset, load_metric
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from numpy import argmax
import wandb

checkpoint = 'checkpoints/baseline/checkpoint-3715'
epochs = 5.

data = load_dataset('benjaminbeilharz/empathetic_dialogues_for_lm')
model = GPT2LMHeadModel.from_pretrained(checkpoint)
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenizer.pad_token = tokenizer.eos_token
wandb.init(project='ba-thesis', entity='benjaminbeilharz')


def tokenize_fn(batch):
    convs = batch['conv']
    dialog = ' '.join([u for u in convs]) + tokenizer.eos_token
    return tokenizer(dialog,
                     return_tensors='pt',
                     truncation=True,
                     padding='max_length')


def compute_metrics(pred):
    metric_dict = {}
    perp = load_metric('perplexity')
    bleu = load_metric('bleu')
    meteor = load_metric('meteor')
    bertscore = load_metric('bertscore')
    logits, labels = pred
    predictions = argmax(logits, axis=-1)
    for metric in [perp, bleu, meteor, bertscore]:
        metric_dict.update(
            metric.compute(predictions=predictions, references=labels))

    pprint(metric_dict)
    #return metric_dict
    return meteor.compute(predictions=predictions, references=labels)


data = data.map(tokenize_fn, batched=False).remove_columns('conv')

args = TrainingArguments(
    'checkpoints/baseline',
    num_train_epochs=epochs,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=10,
    logging_steps=1,
    fp16=True,
    report_to='wandb')
trainer = Trainer(model,
                  args,
                  train_dataset=data['train'],
                  eval_dataset=data['validation'],
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics)

if __name__ == '__main__':
    trainer.predict(data['test'])
