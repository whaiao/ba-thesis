from functools import partial
from typing import Callable, Dict

from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup

from src.models.bert.data import create_next_turn_prediction_dataset

wandb.init(project='ba-thesis', entity='benjaminbeilharz')

# make sure data is in the right format
create_next_turn_prediction_dataset = partial(
    create_next_turn_prediction_dataset, tokenizer=None, is_t5=True)

accelerator = Accelerator()
train, test, val = create_next_turn_prediction_dataset(batch_size=8)
EPOCHS = 10

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
tokenizer = T5Tokenizer.from_pretrained(f't5-base')
model = T5ForConditionalGeneration.from_pretrained(f't5-base').to(device)
optimizer = AdamW(model.parameters(), lr=3e-4)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=EPOCHS *
                                            len(train))

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



for epoch in range(1, EPOCHS + 1):
    best = None
    running_loss = 0.0
    model.train()
    for i, sample in enumerate(tqdm(train), start=1):
        x, y = sample.values()
        inputs = encode(x, y, tokenizer)
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


    eval_running_loss = 0.0
    model.eval()
    for i, sample in enumerate(tqdm(val), start=1):
        with torch.no_grad():
            x, y = sample.values()
            inputs = encode(x, y, tokenizer)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            loss = model(**inputs).loss
            eval_running_loss += loss.item()

        if i % 50 == 0:
            print(
                f'Evaluation: Epoch: {epoch} - Step: {i} - Loss: {loss.item()}'
            )
            wandb.log({'eval/loss': loss,
                'eval/perplexity': torch.exp(loss)})

    avg_train_loss = running_loss / len(train)
    avg_eval_loss = eval_running_loss / len(val)
    print(
        f'Epoch: {epoch} - Average Training Loss: {avg_train_loss} - Average Validation Loss: {avg_eval_loss}'
    )

    if best is None or avg_eval_loss < best:
        best = avg_eval_loss
        print('Saving model checkpoint')
        model.save_pretrained('checkpoints/t5_next_turn_type')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'checkpoints/turn_prediction.pt')
