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

class KnowledgeLoader():
    '''
    Bert Data Loader：
    ALL the DATA FILE should be in the sequence as article, question, option_...., answer.
    '''
    def __init__(self, root_path, train_path, dev_path, test_path, bert_vocab_path,
                batch_size = 8, dev_batch_size = 4, isBertCat = False,
                choice_number = 4, isSkipHead = True, isFixLength = (None, None), isShuffle = True):
        
        self.BertVocabPATH = config.bert_root_dir + bert_vocab_path
        self.tokenizer = BertTokenizer.from_pretrained(self.BertVocabPATH)
        self.isBertCat = isBertCat
        self.choice_number = choice_number
        self.concept_fix_length, self.es_fix_length = isFixLength

        self.dataset_field, self.bert_field = self.init_field(isFixLength)
        self.train_set, self.dev_set, self.test_set = self.init_dataset(root_path, train_path, dev_path, test_path, isSkipHead)
        self.train_iter, self.dev_iter, self.test_iter = self.init_iterator(self.train_set, self.dev_set, self.test_set, batch_size, dev_batch_size, isShuffle)

    def get_train_iter(self):
        if self.isBertCat:
            data_generator = BatchWrapper(self.train_iter, self.field_name[1:], 'label')
        else:
            data_generator = BatchWrapper(self.train_iter, self.field_name, 'label')
        return data_generator

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(np.array(ids.cpu()))

    def get_test_iter(self):
        if self.isBertCat:
            data_generator = BatchWrapper(self.test_iter, self.field_name[1:], 'label')
        else:
            data_generator = BatchWrapper(self.test_iter, self.field_name, 'label')
        return data_generator
    
    def get_dev_iter(self):
        if self.isBertCat:
            data_generator = BatchWrapper(self.dev_iter, self.field_name[1:], 'label')
        else:
            data_generator = BatchWrapper(self.dev_iter, self.field_name, 'label')
        return data_generator
    
    def dataset_len(self):
        return len(self.train_set), len(self.test_set), len(self.dev_set)

    def init_field(self, fix_length):
        sentence_tokenizer = lambda x : self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))
        answer_map = {'A': 0, 'B': 1, 'C': 2, 'D':3 }
        answer_label = lambda x: answer_map[x]

        cls_sentence = Field(sequential=True, tokenize=sentence_tokenizer, use_vocab=False, pad_token=0, batch_first = True)
        cls_concept = Field(sequential=True, tokenize=sentence_tokenizer, use_vocab=False, pad_token=0, batch_first = True, fix_length = self.concept_fix_length)
        cls_es = Field(sequential=True, tokenize=sentence_tokenizer, use_vocab=False, pad_token=0, batch_first = True, fix_length = self.es_fix_length)

        bert_sentence = Field(sequential=True, tokenize=sentence_tokenizer, use_vocab=False, pad_token=0, batch_first = True)
        bert_concept = Field(sequential=True, tokenize=sentence_tokenizer, use_vocab=False, pad_token=0, batch_first = True, fix_length = self.concept_fix_length)
        bert_es = Field(sequential=True, tokenize=sentence_tokenizer, use_vocab=False, pad_token=0, batch_first = True, fix_length = self.es_fix_length)
        
        LABEL = Field(sequential=False, preprocessing = answer_label, use_vocab=False)
        bert_LABEL = Field(sequential=False, use_vocab=False)

        self.field_name, dataset_field, bert_field = ['question'], [('question', cls_sentence)], []
        for i in range(self.choice_number):
            self.field_name.append('option_'+str(i))
            dataset_field.append(('option_'+str(i), cls_sentence))
            bert_field.append(('option_'+str(i), bert_sentence))

            for j in range(9):
                self.field_name.append('option_'+str(i)+'_concept_'+str(j))
                dataset_field.append(('option_'+str(i)+'_concept_'+str(j), cls_concept))
                bert_field.append(('option_'+str(i)+'_concept_'+str(j), bert_concept))
            for j in range(10):
                self.field_name.append('option_'+str(i)+'_es_'+str(j))
                dataset_field.append(('option_'+str(i)+'_es_'+str(j), cls_es))
                bert_field.append(('option_'+str(i)+'_es_'+str(j), bert_es))

        dataset_field.append(('label', LABEL))
        bert_field.append(('label', bert_LABEL))
        return dataset_field, bert_field

 
    def init_dataset(self, root_path, train_path, dev_path, test_path, isSkipHead):
        if self.isBertCat:
            if train_path and dev_path and test_path:
                return BertTabularDataset_MultipleChoice.splits(
                    path = root_path,
                    train = train_path, validation = dev_path, test = test_path,
                    format='tsv',
                    question_fix_length = 40,
                    fields=self.dataset_field,
                    bert_fields=self.bert_field,
                    skip_header=isSkipHead
                )
            else:
                return BertTabularDataset_MultipleChoice(
                    path = os.path.join(root_path, train_path), 
                    format='tsv',
                    question_fix_length = 40,
                    fields=self.dataset_field,
                    bert_fields=self.bert_field,
                    skip_header=isSkipHead
                ), None, None
        else:
            if train_path and dev_path and test_path:
                return TabularDataset.splits(
                    path = root_path,
                    train = train_path, validation = dev_path, test = test_path,
                    format='tsv',
                    fields=self.dataset_field,
                    skip_header=isSkipHead
                )
            else:
                return TabularDataset(
                    path = os.path.join(root_path, train_path), 
                    format='tsv',
                    fields=self.dataset_field,
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

    def __init__(self, path, format, question_fix_length, fields, bert_fields, skip_header=False,
                 csv_reader_params={}, **kwargs):
        format = format.lower()
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
            bert_data = []
            for d in examples:
                question = getattr(d, 'question')
                question = question[-(min(len(question), question_fix_length)):]
                cat_d = []
                for name in bert_fields[:-1]:
                    cat_d.append([101] + question + [102] + getattr(d, name[0]) )
                    
                bert_data.append(cat_d)
       
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