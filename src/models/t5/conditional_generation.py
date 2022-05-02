from pprint import pprint
from functools import partial
from typing import *

from accelerate import Accelerator
from datasets import load_dataset, load_metric
import torch 
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup

from src_old.models.bert.data import create_next_turn_prediction_dataset

wandb.init(project='ba-thesis', entity='benjaminbeilharz')

# make sure data is in the right format
create_next_turn_prediction_dataset = partial(
    create_next_turn_prediction_dataset, tokenizer=None, is_t5=True)

accelerator = Accelerator()
#train, test, val = create_next_turn_prediction_dataset(batch_size=8)
dataset = load_dataset('benjaminbeilharz/ed-for-lm')
train = dataset['train']
val = dataset['validation']
EPOCHS = 3

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
tokenizer = T5Tokenizer.from_pretrained(f't5-base')
model = T5ForConditionalGeneration.from_pretrained(f't5-base').to(device)
optimizer = AdamW(model.parameters(), lr=3e-4)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=500,
                                            num_training_steps=EPOCHS *
                                            len(train))

bleuscore = load_metric('bleu')
meteorscore = load_metric('meteor')
bertscorer = load_metric('bertscore')

wandb.watch(model, log_freq=100)


def encode(x: str, y: str, tokenizer: Callable) -> Dict[str, torch.LongTensor]:
    encoding = tokenizer(x,
                         padding='longest',
                         max_length=512,
                         truncation=True,
                         return_tensors='pt')
    target_encoding = tokenizer(y,
                                padding='longest',
                                max_length=128,
                                truncation=True,
                                return_tensors='pt')

    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
    labels = target_encoding.input_ids
    # ignore pad tokens in labels
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def metrics(generation, response, train: bool = True):
    def _prepare_eval(
        generation: Union[str, List[str]], response: str
    ) -> Mapping[str, Union[List[str], List[List[str]]]]:
        gens = [gen.split() for gen in generation]
        return {
            'prediction': gens,
            'reference':
            [[response.split()] for _ in range(len(gens))]
        }

    bleuscore.add(**_prepare_eval(generation, response))
    meteorscore.add(**_prepare_eval(generation, response))
    bertscorer.add(**_prepare_eval(generation, response))

    bleu = bleuscore.compute()
    meteor = meteorscore.compute()
    bertscore = bertscorer.compute(lang='en')

    metric_dict = {
        'bleu1': bleu['precisions'][0],
        'bleu2': bleu['precisions'][1],
        'bleu3': bleu['precisions'][2],
        'bleu4': bleu['precisions'][-1],
        'bleu': bleu['bleu'],
        'meteor': meteor['meteor'],
        'bertscore-prec': sum(bertscore['precision']) / len(bertscore['precision']),
        'bertscore-recall': sum(bertscore['recall']) / len(bertscore['recall']),
        'bertscore-f1': sum(bertscore['f1']) / len(bertscore['f1'])
    }
    if train:
        metric_dict = {'train/'+k: v for k, v in metric_dict.items()}
    else:
        metric_dict = {'eval/'+k: v for k, v in metric_dict.items()}

    pprint(metric_dict)

    wandb.log(metric_dict)

for epoch in range(1, EPOCHS + 1):
    best = None
    running_loss = 0.0
    model.train()
    for i, sample in enumerate(tqdm(train), start=1):
        x, y, z = sample.values()
        inputs = encode(y, z, tokenizer)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        optimizer.zero_grad()
        loss = model(**inputs).loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        if i % 50 == 0:
            print(f'Epoch: {epoch} - Step: {i} - Loss: {loss.item()}')
            wandb.log({'train/loss': loss,
                'train/perplexity': torch.exp(loss)})
            with torch.no_grad():
                ids = inputs['input_ids']
                gen = model.generate(ids, do_sample=True, top_k=50, top_p=.95, num_return_sequences=1, no_repeat_ngram_size=2, min_length=10, max_length=100)
                decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
                metrics(decoded, z, train=True)

    eval_running_loss = 0.0
    model.eval()
    for i, sample in enumerate(tqdm(val), start=1):
        with torch.no_grad():
            x, y, z = sample.values()
            inputs = encode(y, z, tokenizer)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            loss = model(**inputs).loss
            eval_running_loss += loss.item()

        if i % 50 == 0:
            print(
                f'Evaluation: Epoch: {epoch} - Step: {i} - Loss: {loss.item()}'
            )
            wandb.log({'eval/loss': loss,
                'eval/perplexity': torch.exp(loss)})
            ids = inputs['input_ids']
            gen = model.generate(ids, do_sample=True, top_k=50, top_p=.95, num_return_sequences=1, no_repeat_ngram_size=2, min_length=10, max_length=100)
            decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
            metrics(decoded, z, train=False)

    avg_train_loss = running_loss / len(train)
    avg_eval_loss = eval_running_loss / len(val)
    print(
        f'Epoch: {epoch} - Average Training Loss: {avg_train_loss} - Average Validation Loss: {avg_eval_loss}'
    )

    if best is None or avg_eval_loss < best:
        best = avg_eval_loss
        print('Saving model checkpoint')
        model.save_pretrained('checkpoints/t5-baseline')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'checkpoints/t5-baseline.pt')
