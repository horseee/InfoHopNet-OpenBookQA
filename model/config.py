import os

# Dataset Origin Path
dataset_origin_path = '/data/mxy/nlp/OpenBookQA-V1-Sep2018/'
corpus_path = dataset_origin_path + 'Data/Main/openbook.txt'
train_json_path = dataset_origin_path + 'Data/Main/train.jsonl'
test_json_path = dataset_origin_path + 'Data/Main/test.jsonl'
dev_json_path = dataset_origin_path + 'Data/Main/dev.jsonl'

# Dataset PATH
dataset_path  = 'data'
train_path = 'train_info.tsv'
test_path = 'test_info.tsv'
dev_path = 'dev_info.tsv'

# tsv info
field_name = ['A', 'A_passage', 'B', 'B_passage', 'C','C_passage', 'D', 'D_passage']
label_name = 'label'

# PARAMETERS FOR TRAINING
epoch = 50
vocab_size = 11014

# checkpoint
ckpt_path = 'ckpt/'

# bert
bert_root_dir = '/data/mxy/nlp/bert/'
