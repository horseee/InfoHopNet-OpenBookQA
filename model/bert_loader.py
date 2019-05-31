import io
import os
import codecs
import zipfile
import tarfile
import gzip
import shutil
from functools import partial

import numpy as np
import torch
from torchtext.data import Dataset, Example, TabularDataset
from torchtext.data import Field
from torchtext.data import Iterator, BucketIterator
from torchtext.utils import unicode_csv_reader

from pytorch_pretrained_bert import BertTokenizer

import config

class BertLoader():
    '''
    Bert Data Loader：
    ALL the DATA FILE should be in the sequence as article, question, option_...., answer.
    '''
    def __init__(self, root_path, train_path, dev_path, test_path, bert_vocab_path,
                batch_size = 8, dev_batch_size = 4, isBertCat = True,
                choice_number = 4, isSkipHead = True, isFixLength = None, isShuffle = True):
        
        self.BertVocabPATH = config.bert_root_dir + bert_vocab_path
        self.tokenizer = BertTokenizer.from_pretrained(self.BertVocabPATH)
        self.choice_number = choice_number
        self.fix_length = isFixLength
        self.isBertCat = isBertCat

        self.cls_dataset_field, self.dataset_field, self.bert_field = self.init_field(isFixLength)
        self.train_set, self.dev_set, self.test_set = self.init_dataset(root_path, train_path, dev_path, test_path, isSkipHead)
        self.train_iter, self.dev_iter, self.test_iter = self.init_iterator(self.train_set, self.dev_set, self.test_set, batch_size, dev_batch_size, isShuffle)

    def get_train_iter(self):
        if self.isBertCat:
            data_generator = BatchWrapper(self.train_iter, self.bert_name[0:self.choice_number], 'label')
        else:
            data_generator = BatchWrapper(self.train_iter, self.field_name, 'label')
        return data_generator

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(np.array(ids.cpu()))

    def get_test_iter(self):
        if self.isBertCat:
            data_generator = BatchWrapper(self.test_iter, self.bert_name[0:self.choice_number], 'label')
        else:
            data_generator = BatchWrapper(self.test_iter, self.field_name, 'label')
        return data_generator
    
    def get_dev_iter(self):
        if self.isBertCat:
            data_generator = BatchWrapper(self.dev_iter, self.bert_name[0:self.choice_number], 'label')
        else:
            data_generator = BatchWrapper(self.dev_iter, self.field_name, 'label')
        return data_generator
    
    def dataset_len(self):
        return len(self.train_set), len(self.test_set), len(self.dev_set)

    def init_field(self, fix_length):
        self.field_name = []
        self.bert_name = []
        for i in range(self.choice_number):
            self.field_name.append('option_'+str(i))
            self.field_name.append('option_'+str(i)+'_article')
            self.bert_name.append('option_'+str(i))

        # for dataset examples field
        para_tokenize = lambda x: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))
        add_cls_tokenizer = lambda x: [101] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x)) + [102]
        answer_map = {'A': 0, 'B': 1, 'C': 2, 'D':3 }
        answer_label = lambda x: answer_map[x]

        # for bert examples first step process
        para_PASSAGE = Field(sequential=True, tokenize=para_tokenize, use_vocab=False, pad_token=0, batch_first = True)
        LABEL = Field(sequential=False, preprocessing = answer_label, use_vocab=False)
        dataset_field = [(name, para_PASSAGE) for name in self.field_name]
        dataset_field.append(('label', LABEL))

        # for bert examples field
        bert_PASSAGE = Field(sequential=True, use_vocab=False, pad_token=0, batch_first = True, fix_length = fix_length)
        bert_LABEL = Field(sequential=False, use_vocab=False)
        bert_multiplechioce_field = [(name, bert_PASSAGE) for name in self.bert_name]
        bert_multiplechioce_field.append(('label', bert_LABEL))

         # for normal process
        cls_not_PASSAGE = Field(sequential=True, tokenize=add_cls_tokenizer, use_vocab=False, pad_token=0, batch_first = True)
        cls_PASSAGE = Field(sequential=True, tokenize=add_cls_tokenizer, use_vocab=False, pad_token=0, batch_first = True, fix_length = fix_length)
        LABEL = Field(sequential=False, preprocessing = answer_label, use_vocab=False)
        cls_dataset_field = [(name, cls_not_PASSAGE) if 'article' not in name else (name,cls_PASSAGE) for name in self.field_name]
        cls_dataset_field.append(('label', LABEL))

        return cls_dataset_field, dataset_field, bert_multiplechioce_field

    def init_dataset(self, root_path, train_path, dev_path, test_path, isSkipHead):
        if self.isBertCat:
            if train_path and dev_path and test_path:
                return BertTabularDataset_MultipleChoice.splits(
                    path = root_path,
                    train = train_path, validation = dev_path, test = test_path,
                    format='tsv',
                    fields=self.dataset_field,
                    bert_fields = self.bert_field,
                    article_fix_length = self.fix_length,
                    skip_header=isSkipHead
                )
            else:
                return BertTabularDataset_MultipleChoice(
                    path = os.path.join(root_path, train_path), 
                    format='tsv',
                    fields=self.dataset_field,
                    bert_fields = self.bert_field,
                    article_fix_length = self.fix_length,
                    skip_header=isSkipHead
                ), None, None
        else:
            if train_path and dev_path and test_path:
                return TabularDataset.splits(
                    path = root_path,
                    train = train_path, validation = dev_path, test = test_path,
                    format='tsv',
                    fields=self.cls_dataset_field,
                    skip_header=isSkipHead
                )
            else:
                return TabularDataset(
                    path = os.path.join(root_path, train_path), 
                    format='tsv',
                    fields=self.cls_dataset_field,
                    skip_header=isSkipHead
                ), None, None

    def init_iterator(self, train_set, dev_set, test_set, batch_size, dev_batch_size, isShuffle):
        if dev_set and test_set:
            return BucketIterator.splits(
                [train_set, dev_set, test_set],  
                batch_sizes = (batch_size, dev_batch_size, dev_batch_size),
                shuffle = isShuffle,
                sort = False,
                sort_within_batch = False,
                repeat = False,
            )
        else:
            return BucketIterator(
                train_set,  
                batch_size = batch_size,
                shuffle = isShuffle,
                sort = False,
                sort_within_batch = False,
                repeat = False,
            ), None, None

class BertTabularDataset_MultipleChoice(Dataset):
    """
    Defines a Dataset of columns stored in CSV, TSV, or JSON format.
    This Dataset is used for concat question , options and article in the bert require format.
    """

    def __init__(self, path, format, fields, bert_fields, article_fix_length, skip_header=False,
                 csv_reader_params={}, **kwargs):
        format = format.lower()
        self.article_fix_length = article_fix_length
        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromCSV, 'csv': Example.fromCSV}[format]
        make_bert_example = Example.fromlist

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            if format == 'csv':
                reader = unicode_csv_reader(f, **csv_reader_params)
            elif format == 'tsv':
                reader = unicode_csv_reader(f, delimiter='\t', **csv_reader_params)
            else:
                reader = f

            if format in ['csv', 'tsv'] and isinstance(fields, dict):
                if skip_header:
                    raise ValueError('When using a dict to specify fields with a {} file,'
                                     'skip_header must be False and'
                                     'the file must have a header.'.format(format))
                header = next(reader)
                field_to_index = {f: header.index(f) for f in fields.keys()}
                make_example = partial(make_example, field_to_index=field_to_index)

            if skip_header:
                next(reader)
            
            examples = [make_example(line, fields) for line in reader]
            bert_data = [[[101] + getattr(d, option) + [102] + getattr(d, option+'_article')[0:min(self.article_fix_length, len(getattr(d, option+'_article')))] + [102] 
                            for option in ['option_0', 'option_1', 'option_2', 'option_3']]
                            for d in examples]
           
            _ = [a.append(l.label) for a, l in zip(bert_data, examples)]
            bert_examples = [make_bert_example(data, bert_fields) for data in bert_data]
            
        #  not deal with bert situation
        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(BertTabularDataset_MultipleChoice, self).__init__(bert_examples, bert_fields, **kwargs)

# https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
class BatchWrapper:
    def __init__(self, dl, var, label):
        self.dl, self.var, self.label = dl, var, label
      
    def __iter__(self):
        for batch in self.dl:
            label = getattr(batch, self.label).cuda()

            if self.var is not None:
                data = [getattr(batch, feat).cuda() for feat in self.var]
            else:
                raise NotImplementedError

            yield (data, label)

    def __len__(self):
        return len(self.dl)