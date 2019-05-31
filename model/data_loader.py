import os
import codecs
import numpy as np
import torch

from torchtext.data import TabularDataset
from torchtext.data import Field
from torchtext.data import Iterator, BucketIterator
import torch

class DataLoader():
    def __init__(self, root_path, train_path, dev_path, test_path, 
                field_name, label_name = 'label', batch_size = 8, dev_batch_size = 4,
                isPara = True, isSkipHead = True, isFixLength = None, isShuffle = True):
        self.field_name = field_name
        self.label_name = label_name

        self.dataset_field = self.init_field(field_name, label_name, isPara, isFixLength)
        self.train_set, self.dev_set, self.test_set = self.init_dataset(root_path, train_path, dev_path, test_path, isSkipHead)
        self.train_iter, self.dev_iter, self.test_iter = self.init_iterator(self.train_set, self.dev_set, self.test_set, batch_size, dev_batch_size, isShuffle)

    def get_train_iter(self):
        data_generator = BatchWrapper(self.train_iter, self.field_name, self.label_name)
        return data_generator

    def get_test_iter(self):
        data_generator = BatchWrapper(self.test_iter, self.field_name, self.label_name)
        return data_generator
    
    def get_dev_iter(self):
        data_generator = BatchWrapper(self.dev_iter, self.field_name, self.label_name)
        return data_generator
    
    def dataset_len(self):
        return len(self.train_set), len(self.test_set), len(self.dev_set)

    def init_field(self, field_name, label_name, isPara, fix_length):
        para_tokenize = lambda x: [int(index) for index in x.split()]
        seq_tokenize = lambda x: [[int(index) for index in seq.split(' ')] for seq in x.split(',')]
        answer_map = {'A': 0, 'B': 1, 'C': 2, 'D':3 }
        answer_label = lambda x: answer_map[x]

        para_PASSAGE = Field(sequential=True, tokenize=para_tokenize, use_vocab=False, pad_token=0, batch_first = True)
        para_fix_PASSAGE = Field(sequential=True, tokenize=para_tokenize, use_vocab=False, pad_token=0, batch_first = True, fix_length = fix_length)
        seq_PASSAGE = Field(sequential=True, tokenize=seq_tokenize, use_vocab=False, pad_token=0, batch_first = True)
        LABEL = Field(sequential=False, preprocessing = answer_label, use_vocab=False)

        dataset_field = []
        for name in field_name:
            if '_' in name:
                dataset_field.append((name, para_fix_PASSAGE))
            else:
                dataset_field.append((name, para_PASSAGE))
        dataset_field.append((label_name, LABEL))
      
        return dataset_field

    def init_dataset(self, root_path, train_path, dev_path, test_path, isSkipHead):
        return TabularDataset.splits(
            path = root_path,
            train = train_path, validation = dev_path, test = test_path,
            format='tsv',
            skip_header=isSkipHead, 
            fields=self.dataset_field
        )

    def init_iterator(self, train_set, dev_set, test_set, batch_size, dev_batch_size, isShuffle):
        return BucketIterator.splits(
            [train_set, dev_set, test_set],  
            batch_sizes = (batch_size, dev_batch_size, dev_batch_size),
            shuffle = isShuffle,
            sort = False,
            sort_within_batch = False,
            repeat = False,
        )

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


'''
def read_index_data(data_path):
    question_set = []
    answer_map = {'A': 0, 'B': 1, 'C': 2, 'D':3 }
    
    with codecs.open(data_path, 'r', 'utf-8') as file:
        for line in file:
            choices_list = line.strip().split('\t')
            choice_pre_hyp = []
            for i in range(4):
                text = [int(index) for index in choices_list[i * 2].split(' ')]
                if choices_list[i * 2 + 1] == '':
                    cor = [2]
                else:
                    cor = [int(index) for index in choices_list[i * 2 + 1].split(' ')]
                choice_pre_hyp.append([text, cor])
            answer = answer_map[choices_list[-1]]
            choice_pre_hyp.append(answer)
            question_set.append(choice_pre_hyp)
    return np.array(question_set)

def pad_sequence(sequences):
    #pad_data = nn.utils.rnn.pad_sequence(sequences)
    sequence_length = [len(seq) for seq in sequences]
    len_max = max(sequence_length)
    pad_data = [x + [0] * (len_max - len(x)) for x in sequences]
    return torch.tensor(pad_data, dtype=torch.long)

def batch_generator(dataset_path, batch_size):
    dataset =  read_index_data(dataset_path)
    data_size = len(dataset)
    np.random.shuffle(dataset)
    batch_num = data_size // batch_size
    for i in range(batch_num):
        questions = dataset[batch_size * i:batch_size * (i + 1)]
        choice_pre_hyp = []
        for i in range(4):
            text = pad_sequence([question[i][0] for question in questions]).cuda()
            cor = pad_sequence([question[i][1] for question in questions]).cuda()
            choice_pre_hyp.append([text, cor])
        answer = torch.tensor([question[4] for question in questions]).cuda()
        yield choice_pre_hyp,answer
'''