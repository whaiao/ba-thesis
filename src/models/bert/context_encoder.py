from typing import *

from datasets import load_dataset, load_metric
from more_itertools import pairwise

from tqdm import tqdm
from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup
import torch
import wandb



def _sample(sample: Mapping[str, List[str]]) -> Tuple[str, Iterable]:
    sample = sample['conv']
    ctx = sample.pop(0)
    iterable = pairwise(sample)
    return (ctx, iterable)

def _prepare_dialog_history(ctx: str,
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


data = load_dataset('benjaminbeilharz/empathetic_dialogues_for_lm')
device = torch.device('cuda')
epochs = 5
total_train_steps = int(epochs) * len(data['train'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased').to(device)
bert2bert.config.decoder.is_decoder = True
bert2bert.config.decoder.add_cross_attention = True
bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
bert2bert.config.eos_token_id = tokenizer.eos_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id

bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size
bert2bert.config.max_length = 142
bert2bert.config.min_length = 56
bert2bert.config.no_repeat_ngram_size = 3
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 4

optim = AdamW(bert2bert.parameters())
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=500,
        num_training_steps=total_train_steps)

def train():
    wandb.init(entity='benjaminbeilharz', project='ba-thesis')
    wandb.watch(bert2bert, log_freq=100)
    bert2bert.zero_grad()

    for epoch in range(1, epochs+1):
        bert2bert.train()
        for i, sample in enumerate(tqdm(data['train']), start=1):
            ctx, dialog = _sample(sample)
            dialog = _prepare_dialog_history(ctx, dialog)

            for turn in dialog:
                history, utterance, _ = turn
                enc = tokenizer(history, truncation=True, return_tensors='pt').to(device)
                dec = tokenizer(utterance, truncation=True, return_tensors='pt').to(device)

                # forward
                optim.zero_grad()
                bert2bert.zero_grad()
                out = bert2bert(input_ids=enc.input_ids, attention_mask=enc.attention_mask, decoder_input_ids=dec.input_ids, decoder_attention_mask=dec.attention_mask, labels=dec.input_ids)
                loss = out.loss
                logits = out.logits
                loss /= len(dialog)
                # backprop
                loss.backward()
                optim.step()
                scheduler.step()
                if i % 50 == 0:
                    pred = torch.argmax(logits, dim=-1)[0]
                    print('Training generation')
                    print(f'Generation at epoch: {epoch}\nStep:\t{i}')
                    print(tokenizer.decode(pred))

            wandb.log({
                'train/loss': loss.item(),
                'train/perplexity': torch.exp(loss),
                'train/batch_progress': i/len(data['train'])
                })

        
        print('saving bert2bert after epoch {}'.format(epoch))
        bert2bert.save_pretrained(f'checkpoints/bert2bert-empathetic-dialogues/epoch-{epoch}')
        tokenizer.save_pretrained(f'checkpoints/bert2bert-empathetic-dialogues/epoch-{epoch}')

        # validation
        bert2bert.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(data['validation']), start=1):
                ctx, dialog = _sample(sample)
                dialog = _prepare_dialog_history(ctx, dialog)

                for turn in dialog:
                    history, utterance, _ = turn
                    enc = tokenizer(history, truncation=True, return_tensors='pt').to(device)
                    dec = tokenizer(utterance, truncation=True, return_tensors='pt').to(device)

                # forward
                    out = bert2bert(input_ids=enc.input_ids, attention_mask=enc.attention_mask, decoder_input_ids=dec.input_ids, decoder_attention_mask=dec.attention_mask, labels=dec.input_ids)
                    loss = out.loss
                    logits = out.logits
                    if i % 50 == 0:
                        pred = torch.argmax(logits, dim=-1)[0]
                        print('Validation generation')
                        print(f'Generation at epoch: {epoch}\nStep:\t{i}')
                        print(tokenizer.decode(pred))
                    loss /= len(dialog)

                wandb.log({
                    'validation/loss': loss.item(),
                    'validation/perplexity': torch.exp(loss),
                    'validation/batch_progress': i/len(data['validation'])
                    })


train()
