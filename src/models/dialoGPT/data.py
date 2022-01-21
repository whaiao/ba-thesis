from collections import defaultdict
import re
from typing import List, Union

import pandas as pd

from datasets.dataset_dict import DatasetDict
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

import torch
from torch.utils.data import Dataset, DataLoader

HuggingfaceDataset = DatasetDict


class DialoGPTDataset(Dataset):
    def __init__(
        self,
        data: HuggingfaceDataset,
        n_turns: int,
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            'microsoft/DialoGPT-large')):
        super(DialoGPTDataset, self).__init__()
        self.dataset = data
        self.n_turns = n_turns
        self.tokenizer = tokenizer

        self.prepare_dataframe()
        self.construct_data()

    @classmethod
    def empathetic_dialogues(cls, n_turns: int):
        pattern = re.compile(r'_conv:\d+')
        data = load_dataset('empathetic_dialogues')
        datasets = {}
        for k, split in data.items():
            convs = defaultdict(list)
            prompts = {}
            labels = {}

            for i, sample in enumerate(split):
                conv_id = re.sub(pattern, '', sample['conv_id'])
                convs[conv_id].append(sample['utterance'])
                prompts[conv_id] = sample['prompt']
                labels[conv_id] = sample['context']

            for i, j in prompts.items():
                convs[i].insert(1, j)

            datasets[k] = list(convs.values())

        return cls(data=datasets, n_turns=n_turns)

    def prepare_dataframe(self):
        columns = ['response', 'context']
        columns = columns + ['context/' + str(i) for i in range(self.n_turns)]
        dialogues = []
        for split, dataset in self.dataset.items():
            for sample in dataset:
                row = [
                    turn for i, turn in enumerate(sample)
                    if i < self.n_turns + 2
                ]
                dialogues.append(row)

        df = pd.DataFrame.from_records(dialogues, columns=columns)
        print(df.head())
        self.df = df

    def construct_data(self):
        def construct_conv(row: List[str]) -> List[int]:
            return [
                self.tokenizer.encode(x) + [self.tokenizer.eos_token_id]
                for x in row if x is not None
            ]

        self.data = []
        for _, row in self.df.iterrows():
            conv = construct_conv(row)
            print(conv)
            self.data.append(conv)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


if __name__ == "__main__":
    dataset = DialoGPTDataset.empathetic_dialogues(n_turns=7)
