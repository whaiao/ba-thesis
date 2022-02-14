from more_itertools import pairwise
from pprint import pprint
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding
import datasets
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
import pandas as pd
from transformers.tokenization_utils import PreTrainedTokenizer

HuggingfaceDataset = DatasetDict
TOKENIZER = AutoTokenizer.from_pretrained('bert-base-uncased')


class DailyDialogDataset(Dataset):
    def __init__(self, data: HuggingfaceDataset, tokenizer=TOKENIZER):
        super(DailyDialogDataset, self).__init__()
        self.dataset = {}
        for k, v in data.items():
            print(k)
            self.dataset[k] = self.flatten_samples(v)

        self.dataset = datasets.DatasetDict(self.dataset)
        self.tokenizer = tokenizer

        self.label_dict = {
            'dummy': 0,
            'inform': 1,
            'question': 2,
            'directive': 3,
            'commissive': 4
        }

    @classmethod
    def load_from_huggingface(
            cls,
            split: str,
            path: str = 'benjaminbeilharz/better_daily_dialog'):
        data = load_dataset(path, split=split)
        return cls(data=data)

    def flatten_samples(self, split):
        self.data = []
        self.labels = []
        self.emotions = []
        self.dialog_id = []
        dialogs = split['dialog']
        labels = split['act']
        emotions = split['emotion']

        for i, (dialog, label,
                emotion) in enumerate(zip(dialogs, labels, emotions)):
            for j, _ in enumerate(dialog):
                self.data.append(dialog[j])
                self.labels.append(label[j])
                self.emotions.append(emotion[j])
                self.dialog_id.append(i)

        return datasets.Dataset.from_dict({
            'dialog_id': self.dialog_id,
            'utterance': self.data,
            'turn_type': self.labels,
            'emotion': self.emotions
        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self,
                    idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        tokens = self.tokenizer(self.data[idx],
                                padding=True,
                                truncation=True,
                                return_tensors='pt')

        tokens['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        print(tokens)
        return tokens

    @staticmethod
    def create_dataloaders(batch_size: int, collate_fn) -> List[DataLoader]:
        dataloaders = []
        for split in ['train', 'validation', 'test']:
            dataset = DailyDialogDataset.load_from_huggingface(split)
            dataloaders.append(
                DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           collate_fn=collate_fn))

        return dataloaders


def create_datasets(tokenizer: PreTrainedTokenizer):
    TOKENIZER = tokenizer

    def tokenizer_fn(sample):
        """This takes one sample and adds it to all datasets.
        Data Collator only takes arguments which are suitable for the
        underlying Transformer
        """
        return TOKENIZER(sample['utterance'], truncation=True)

    data = load_dataset('benjaminbeilharz/better_daily_dialog')
    tokenized = data.map(tokenizer_fn, batched=True)
    tokenized = tokenized.remove_columns(['dialog_id', 'emotion', 'utterance'])
    tokenized = tokenized.rename_column('turn_type', 'labels')

    return tokenized


def create_next_turn_prediction_dataset(
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 1,
        is_t5: bool = False) -> Union[DatasetDict, List[datasets.Dataset]]:
    """Creates a dataset for next dialog turn prediciton

    Args:
        tokenizer - tokenizer to use
        batch_size - batch size of samples (default 1)
        is_t5 - if `True` returns `DataLoader` classes of splits

    Returns:
        daily dialog dataset prepared for next turn prediction
    """
    TOKENIZER = tokenizer

    def tokenizer_fn(sample):
        return TOKENIZER(sample['first'], sample['second'], truncation=True)

    # templates to tune t5 for
    templates = {
        1: 'inform: ',
        2: 'ask a question: ',
        3: 'make a suggestion: ',
        4: 'accept a suggestion: '
    }

    def relabel_samples(sample):
        new_data = []
        for s in sample:
            turns = s['dialog']
            labels = s['act']

            for i, turn in enumerate(pairwise(turns), start=1):
                x, y = turn

                if not is_t5:
                    new_data.append({
                        'first': x,
                        'second': y,
                        'labels': labels[i]
                    })
                else:
                    x = templates[labels[i]] + x
                    new_data.append({'first': x, 'second': y})

        return new_data

    data = load_dataset('daily_dialog')
    final = DatasetDict()
    for k, v in data.items():
        tmp = pd.DataFrame(relabel_samples(v))
        final[k] = datasets.Dataset.from_pandas(tmp)

    if is_t5:
        # return pytorch dataset if using data for t5
        t5 = []
        for split in final.keys():
            t5.append(DataLoader(final[split], batch_size=batch_size))
        return t5

    tokenized = final.map(tokenizer_fn, batched=True)
    tokenized.remove_columns_(['first', 'second'])
    return tokenized
