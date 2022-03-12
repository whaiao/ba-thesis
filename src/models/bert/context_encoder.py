from typing import *

from datasets import load_dataset, load_metric
from more_itertools import pairwise

from tqdm import tqdm
from transformers import BertGenerationEncoder, BertGenerationDecoder, BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments, DataCollatorWithPadding, AdamW, get_linear_schedule_with_warmup
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

device = torch.device('cuda')
epochs = 5

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data = load_dataset('benjaminbeilharz/empathetic_dialogues_for_lm')

encoder = BertGenerationEncoder.from_pretrained('bert-base-uncased')
decoder = BertGenerationDecoder.from_pretrained('bert-base-uncased',
        add_cross_attention=True,
        is_decoder=True)


total_train_steps = int(epochs) * len(data['train'])
bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder).to(device)
optim = AdamW(bert2bert.parameters())
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0,
        num_training_steps=total_train_steps)

bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id
bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size


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
                input_ids = tokenizer(history, add_special_tokens=False, truncation=True, return_tensors='pt').input_ids.to(device)
                labels = tokenizer(utterance, truncation=True, return_tensors='pt').input_ids.to(device)

                # forward
                loss = bert2bert(input_ids=input_ids, labels=labels).loss
                loss /= len(dialog)
                loss.backward()

            wandb.log({
                'train/loss': loss.item(),
                'train/perplexity': torch.exp(loss),
                'train/batch_progress': i/len(data['train'])
                })

            # backprop
            optim.step()
            scheduler.step()
            optim.zero_grad()
        
        print('saving bert2bert after epoch {}'.format(epoch))
        bert2bert.save_pretrained(f'checkpoints/bert2bert-empathetic-dialogues/epoch-{epoch}')

        # validation
        bert2bert.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(data['validation']), start=1):
                ctx, dialog = _sample(sample)
                dialog = _prepare_dialog_history(ctx, dialog)

                for turn in dialog:
                    history, utterance, _ = turn
                    input_ids = tokenizer(history, add_special_tokens=False, truncation=True, return_tensors='pt').input_ids.to(device)
                    labels = tokenizer(utterance, truncation=True, return_tensors='pt').input_ids.to(device)

                    # forward
                    loss = bert2bert(input_ids=input_ids, labels=labels).loss
                    loss /= len(dialog)

                wandb.log({
                    'validation/loss': loss.item(),
                    'validation/perplexity': torch.exp(loss)
                    })

            generation = bert2bert.generate(input_ids)
            decoded = tokenizer.decode(generation[0], skip_special_token=True)

        print(f'Input context: {history}')
        print('Input token_ids: ', input_ids)
        print('Generated token ids: ', generation)
        print(f'Generation after epoch: {epoch}\n{decoded}')


train()
