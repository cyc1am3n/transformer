import os

import torch
from torchtext.data import Field, BucketIterator
from torchtext.data import TabularDataset

class DataLoader(object):
    """DataLoader class"""
    def __init__(self, config, tokenize_en, tokenize_fr):
        self.config = config
        self.EN_TEXT = Field(tokenize = tokenize_en,
            batch_first=True,
            tokenizer_language="en",
            init_token = '<s>',
            eos_token = '</s>')
        self.FR_TEXT = Field(tokenize = tokenize_fr,
            batch_first=True,
            tokenizer_language="fr",
            init_token = '<s>',
            eos_token = '</s>')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_train_valid_data(self):
        train_data, valid_data = TabularDataset.splits(path=self.config['dir'],
                                             train=self.config['train'],
                                             validation=self.config['valid'],
                                             format='csv',
                                             fields=[('en', self.EN_TEXT), ('fr', self.FR_TEXT)])
        self.EN_TEXT.build_vocab(train_data, valid_data)
        self.FR_TEXT.build_vocab(train_data, valid_data)
        return train_data, valid_data

    def _get_test_data(self):
        test_data = TabularDataset(path=os.path.join(self.config['dir'], self.config['test']),
                                format='csv',
                                fields=[('en', EN_TEXT), ('fr', FR_TEXT)])
        return test_data

    def get_train_valid_iter(self, batch_size):
        train_data, valid_data = self._get_train_valid_data()
        train_iter, valid_iter = BucketIterator.splits(
            (train_data, valid_data),
            batch_size = batch_size,
            device = self.device)
        return train_iter, valid_iter

    def get_test_iter(self, batch_size):
        test_data = self._get_test_data()
        test_iter = BucketIterator(test_data,
            batch_size=batch_size,
            device=self.device)
