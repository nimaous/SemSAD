
# coding=utf-8
# Copyright 2021 The SemSAD Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import logging
import torch
import io
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import ngrams_iterator, get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
import pickle as pk
from word_level_augmentation import word_level_augment, TfIdfWordRep, get_data_stats
from simple_tokenization import FullTokenizer
from tqdm import tqdm
import string
from nltk.corpus import wordnet 


URLS = {
    'AG_NEWS':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms',
    'SogouNews':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE',
    'DBpedia':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k',
    'YelpReviewPolarity':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg',
    'YelpReviewFull':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0',
    'YahooAnswers':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU',
    'AmazonReviewPolarity':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM',
    'AmazonReviewFull':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA'
}


uda_w_tokenizer = FullTokenizer(vocab_file=None)

def thesaurus_susbtitution(tokens, replace_prt):
    res_tokens = []
    for token in tokens:
        if token not in string.punctuation:
            token_syn = ''
            syns = wordnet.synsets(token) 
            if syns:
                token_syn = np.random.choice(np.random.choice(syns).lemmas()).name()
                if replace_prt >= np.random.rand():
                    token = token_syn 
        res_tokens.append(token)
    return res_tokens


def _csv_iterator(data_path, yield_cls=False):
    torch_tokenizer = get_tokenizer("basic_english")
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)        
        for row in reader:
            paragraph = ' '.join(row[1:])
            tokens = uda_w_tokenizer.tokenize_to_word(paragraph)
            #tokens = torch_tokenizer(tokens)
            if yield_cls:
                yield int(row[0]) - 1, tokens
            else:
                yield tokens


def _create_data_from_iterator(vocab, iterator, out_cls): # vocab can be removed 
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        for cls, tokens in iterator:
            if cls not in out_cls:
                if len(tokens) == 0:
                    logging.info('Row contains no tokens.')
                data.append((cls, tokens))
                labels.append(cls)
                t.update(1)
    return data, set(labels)



class TextDataset(Dataset):
    
    def __init__(self, dataset_name, 
                               root='.',
                               vocab= None,
                               train=True, 
                               tf_idf_pr = 0.3,
                               rnd_tf_idf_pr = 0.0, 
                               rnd_th_sus_prt = 0.0,
                               th_sus_prt = 0.5, 
                               data_stats_path = None, #'tf-idf-dict',
                               sufix_name= "",
                               tokens_path = None, #'train_tokens', 
                               out_cls=None,
                               sync_aug=False):
        super(TextDataset ,self).__init__()
        assert isinstance(out_cls, list)==True
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
        extracted_files = extract_archive(dataset_tar)
        self.th_sus_prt = th_sus_prt
        self.rnd_th_sus_prt = rnd_th_sus_prt
        self.sync_aug = sync_aug
        for fname in extracted_files:
            if fname.endswith('train.csv'):
                train_csv_path = fname
            if fname.endswith('test.csv'):
                test_csv_path = fname

    
        if tokens_path:
            print("loading the tokens")
            with open(tokens_path, 'rb') as tokens_f:
                self.data = pk.load(tokens_f) 
        if train :   
            if not tokens_path:
                logging.info('Creating training data')            
                self.data, self.labels = _create_data_from_iterator(vocab, _csv_iterator(train_csv_path,
                                                                                         yield_cls=True, 
                                                                                             ),out_cls=out_cls)
                with open(f'tokens_{sufix_name}' , 'wb') as tokens_f:                
                    pk.dump(self.data, tokens_f)
        else:
            if not tokens_path:
                logging.info('Creating testing data')
                self.data, self.labels = _create_data_from_iterator( vocab, _csv_iterator(test_csv_path, 
                                                                                          yield_cls=True,
                                                                                          ),out_cls=out_cls) 
                with open(f'tokens_{sufix_name}' , 'wb') as tokens_f:                
                    pk.dump(self.data, tokens_f)            
        
        if data_stats_path != None:
            print("Loading tf-idf dictionary ................")
            with open(data_stats_path, 'rb') as stat_path:
                self.data_stats = pk.load(stat_path)
        else:
            print("Creating Data tf-idf dictionary (take a few mintues................")
            self.data_stats = get_data_stats(self.data)
            with open(f'tf-idf-dict_{sufix_name}', 'wb') as stat_path:
                pk.dump(self.data_stats, stat_path)        
                print(f"tf-idf dictionary is created and stored in tf-idf-dict_{sufix_name}")
        self.tfidf_rp = TfIdfWordRep(tf_idf_pr, self.data_stats)   
        self.tfidf_rp_rnd = TfIdfWordRep(rnd_tf_idf_pr, self.data_stats)   
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        label = self.data[index][0]
        
        rnd_idx1 = np.random.randint(len(self.data))
        rnd_idx2 = np.random.randint(len(self.data))
        rnd_idx3 = np.random.randint(len(self.data))
        
        rnd_tokens = self.data[rnd_idx1][1]
        rnd_org_s1 = ' '.join(self.data[rnd_idx2][1])
        rnd_org_s2 = ' '.join(self.data[rnd_idx3][1])

        tokens = self.data[index][1]
        tfidf_tokens1 = self.tfidf_rp(tokens)
        if self.sync_aug:
            ts_tokens = thesaurus_susbtitution(tfidf_tokens1 , 
                                               self.th_sus_prt) 
        else:
            ts_tokens = thesaurus_susbtitution(tokens , 
                                               self.th_sus_prt)  
            
        tfidf_tokens = self.tfidf_rp(tokens)
        tfidf_rnd_tokens = self.tfidf_rp_rnd(rnd_tokens)
        ts_tfidf_tokens = thesaurus_susbtitution(tfidf_tokens , 
                                                 self.th_sus_prt)
        ts_tfidf_rnd_tokens = thesaurus_susbtitution(tfidf_rnd_tokens , 
                                                     self.rnd_th_sus_prt)
        s1 = ' '.join(ts_tokens)
        s2 = ' '.join(ts_tfidf_tokens)
        rnd_text = ' '.join(ts_tfidf_rnd_tokens)
        return s1, s2, rnd_text, rnd_org_s1, rnd_org_s2, label
    
    


        
    
    