import os
import codecs
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from model import match_lstm, rnet, InfoHopNet, MemoryHopNet
from modules import get_segment_ids, get_mask_ids
import config
from data_loader import DataLoader
from bert_loader import BertLoader
from knowledge_loader import KnowledgeLoader
from VisdomPainter import visdom_painter as Painter
from evaluation import eval_model

from pytorch_pretrained_bert import BertModel, BertForMultipleChoice, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

import argparse
parser = argparse.ArgumentParser(description = 'model select')
parser.add_argument('--model', '-m', help = 'model name')
parser.add_argument('--gpuid', '-g', type=int, help = 'gpu id')
parser.add_argument('--lr', '-l', type=float, help="learning rate")
parser.add_argument('--hidden_size', type=int, help='hidden states dimension')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--gradient_accumulation', type=int, help='gradient accumulation')
parser.add_argument('--hop_num', type=int, default=2, help='hop num in attention ranking')
parser.add_argument('--head_num', type=int, default=4, help='head num in multihead attention')
parser.add_argument('--port', type=int, help='visdom port')
parser.add_argument
args = parser.parse_args()

seed = 18100
torch.cuda.set_device(args.gpuid)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class IterOpt:
    def __init__(self, optimizer, rate, iter_decay):
        self.optimizer = optimizer
        self._step = 0
        self._rate = rate
        self.iter_decay = iter_decay
        
    def step(self):
        self._step += 1
        self._rate = self._rate * self.iter_decay
        for p in self.optimizer.param_groups:
            p['lr'] = self._rate
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()

if __name__ == "__main__":
    start_time = time.gmtime()
    batch_size = args.batch_size / args.gradient_accumulation
    print("-----------------------Loading Data-----------------------")
    if args.model == 'InfoHopNet':
        batch_loader = KnowledgeLoader(root_path = config.dataset_path, 
                              train_path = 'train_info.tsv', 
                              dev_path = 'dev_info.tsv', 
                              test_path = 'test_info.tsv',
                              bert_vocab_path = 'bert-base-uncased-vocab.txt',
                              batch_size = batch_size,
                              dev_batch_size = batch_size/2,
                              isSkipHead = False,
                              isBertCat = True,
                              isFixLength = (50, 50)
        )
    elif args.model == 'MemoryHopNet':
        batch_loader = KnowledgeLoader(root_path = config.dataset_path, 
                              train_path = 'train_info.tsv', 
                              dev_path = 'dev_info.tsv', 
                              test_path = 'test_info.tsv',
                              bert_vocab_path = 'bert-base-uncased-vocab.txt',
                              batch_size = batch_size,
                              dev_batch_size = batch_size,
                              isSkipHead = False,
                              isBertCat = False,
                              isFixLength = (30, 30)
        )
    else:
        raise NotImplementedError
 
    trainset_len, testset_len, devset_len = batch_loader.dataset_len()
    print("size of the dataset: train={}, test={}, dev={}".format(trainset_len, testset_len, devset_len))
    
    print("-----------------------Model Structure-----------------------")
    print("Training Model:", args.model)
    if args.model == 'InfoHopNet':
        model = InfoHopNet(
            bert_path = 'bert-base-uncased.tar.gz', 
            bert_dim = 768, hidden_dim = args.hidden_size, 
            hop_num = args.hop_num, 
            head_num = args.head_num,
            dropout_rate = 0.1
        ).cuda()
    elif args.model == 'MemoryHopNet':
        model = MemoryHopNet(
            batch_size = batch_size,
            hidden_dim = 768,
            bert_parameter = 'bert-base-uncased.tar.gz',
            head_num = args.head_num,
            hop_num = args.hop_num
        ).cuda()
    else:
        model = eval(args.model)(batch_size = args.batch_size,
                            num_classes = 3,
                            glove_path = 'data/glove.300d.small_concept.txt',
                            vocab_size = config.vocab_size,
                            input_dim = 300,
                            hidden_dim = args.hidden_size,
                            embedding_dim = 300,
                            entity_path = None, 
                            dropout_rate = 0.1).cuda()
    loss_fn = nn.MultiMarginLoss(margin=0.5).cuda()
    print(model)

    if not os.path.exists('ckpt'):
        os.mkdir('ckpt')

    print("-----------------------Start Training-----------------------")
    std_opt = IterOpt(  optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas = (0.9, 0.999), weight_decay=1e-5),
                        rate = args.lr,
                        iter_decay = 1 )
    
    painter = Painter(port = args.port)
    legend = ['dev', 'test']
    painter.create_window([0], [1.4], "Train loss", title = "Train Loss - " + args.model)
    painter.create_window([0], [0], "Train accuracy",title = "Train Accuracy - " + args.model)
    painter.create_window(torch.zeros((1,)), torch.zeros((1,2)), "accuracy", title = "Valid Accuracy - " + args.model , legends = legend)

    test_epoch = 0
    best_test_accuracy = 0
    step, global_step = 0, 0
    for e in range(config.epoch):
        print("Training: Epoch %d" % e)
        print('Learning rate : {0}'.format(std_opt.optimizer.param_groups[0]['lr']))
        loss_rate, correct = [], 0
            
        for choices, answer in batch_loader.get_train_iter():
            model.train()
            step += 1

            if args.model == 'InfoHopNet':
                data = []
                for i in range(4):
                    start_index = 20 * i
                    choice = [choices[start_index], 
                            torch.stack(choices[start_index + 1: start_index + 10], 1), 
                            torch.stack(choices[start_index + 10: start_index + 15], 1)]
                    data.append(choice)
                output = model(data, is_training = True)[0]
            elif args.model == 'MemoryHopNet':
                texts = []
                for i in range(4):
                    texts.append(torch.stack(choices[i*20 + 2: i*20 + 11], 1))
                output = model(choices[0], choices[1:-1:20], texts)
            else:
                data = []
                for i in range(4):
                    data.append((choices[2 * i], choices[2 * i + 1]))
                output = model(data)
                

            loss = loss_fn(output, answer)
            loss.backward()

            if step % args.gradient_accumulation != 0:
                continue
            global_step += 1

            pred = torch.max(output, 1)[1]
            correct = correct + torch.eq(pred, answer).cpu().sum().item()
            loss_rate.append(loss.cpu().item())
                
            std_opt.step()
            std_opt.zero_grad()
            
            if global_step % 10 == 0:
                loss = np.mean(loss_rate)
                accuracy = round(correct / (10 * choices[0].shape[0]), 4)
                print('[INFO]step {}: accuracy = {}, loss = {}'.format(global_step, accuracy, loss))
                painter.update_data("Train loss", [global_step], [loss])
                painter.update_data("Train accuracy", [global_step], [accuracy])
                correct, loss_rate = 0, []
            
            if global_step % 50 == 0:
                test_epoch += 1
                dev_acc = eval_model(model, batch_loader.get_dev_iter(), devset_len, args.model)
                test_acc = eval_model(model, batch_loader.get_test_iter(), testset_len, args.model)

                print('epoch {} : dev accuracy = {}, test accuracy = {}'.format(test_epoch, dev_acc, test_acc))
                painter.update_data("accuracy", [[test_epoch, test_epoch]], [[dev_acc, test_acc]])

                if test_acc > best_test_accuracy:
                    best_test_accuracy = test_acc
                    torch.save(model, 'ckpt/test-' + args.model + '-' + time.strftime("%H:%M:%S", start_time) + '-bestacc.pb')


        print("Epoch {} training finished!".format(e))
        
        
