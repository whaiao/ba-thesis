from datasets import load_dataset, load_metric
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from numpy import argmax
import wandb

checkpoint = 'checkpoints/baseline/checkpoint-3715'
epochs = 5.

data = load_dataset('benjaminbeilharz/empathetic_dialogues_for_lm')
model = GPT2LMHeadModel.from_pretrained(checkpoint)
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
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

    return metric_dict


data = data.map(tokenize_fn, batched=True).remove_columns('conv')

args = TrainingArguments(
    'checkpoints/baseline',
    num_train_epochs=epochs,
    save_strategy='epoch',
    evaluation_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=3,
    #gradient_checkpointing=True,
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
    trainer.evaluate()
