import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from modules import RNNEncoder, GloveEmbedding, EntityEmbedding, mLstm, MultihopAttention
from modules import MultiheadAttention, AttentionRank, InterAttention
from modules import get_segment_ids, get_mask_ids
import config

from pytorch_pretrained_bert import BertModel

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class match_lstm(torch.nn.Module):
    def __init__(self, batch_size, num_classes,
                glove_path, vocab_size, embedding_dim = 300, 
                input_dim =300, hidden_dim = 150, 
                entity_path = None, rel_size = 0, rel_dim = 3, entity_lambda = [0, 0, 0],
                dropout_rate = 0.1):
        super(match_lstm, self).__init__()
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Embedding
        self.glove_path = glove_path
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = GloveEmbedding(glove_path = self.glove_path, 
                                        vocab_size = self.vocab_size, 
                                        embedding_dim = self.embedding_dim)

        # Entity
        self.entity_path = entity_path
        if self.entity_path != None:
            self.rel_size = rel_size
            self.rel_dim = rel_dim
            self.entity_lambda = entity_lambda
            self.entity_embedding = EntityEmbedding(entity_path = self.entity_path,
                                                    rel_size = self.rel_size,
                                                    vocab_size = self.vocab_size,
                                                    rel_dim = self.rel_dim)
        else:
            self.entity_embedding = None
            self.entity_lambda = None

        # LSTM - Encode
        self.encoder = RNNEncoder(self.batch_size, self.input_dim, self.hidden_dim, self.dropout_rate, 
                                  mode = 'LSTM', bidirectional = False)
        # mLSTM
        self.m_lstm_attention = mLstm(self.hidden_dim, self.hidden_dim, self.batch_size, 
                                      entity_embedding = self.entity_embedding, 
                                      entity_lambda = self.entity_lambda)
        
        # fully connect
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
                nn.Dropout(p = self.dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Dropout(p = self.dropout_rate),
                nn.Linear(self.hidden_dim, 1)
        )
        
    def forward(self, answers):
        
        answer_hiddens = []
        for answer in answers:
            # Glove Embedding and biLSTM Encoder
            text_embed, text_mask = self.embedding(answer[0])
            cor_embed, cor_mask = self.embedding(answer[1])
            text_encode = self.encoder(text_embed, text_mask)
            cor_encode = self.encoder(cor_embed, cor_mask)
            #print(text_encode.shape, cor_encode.shape)
            
            # MatchLSTM
            hidden_last = self.m_lstm_attention(text_encode, cor_encode, answer[0], answer[1])[-1] # seq_len * (batch * hidden_dim)
            #hidden_last = torch.stack(hidden_last, 1) # batch * seq_len * hidden_dim
            #hidden_encode = torch.cat([text_encode - hidden_last, text_encode * hidden_last], -1) # batch * seq_len * (hidden_dim * 2)
            #fuse_mean = torch.mean(hidden_encode, 1) # batch * (hidden_dim * 2)
            #answer_hiddens.append(fuse_mean)
            answer_hiddens.append(hidden_last)
    
        # Fully connect layer
        answer_stack = torch.stack(answer_hiddens, 1)
        predict_op = self.classifier(answer_stack).squeeze(-1)
        return predict_op

class rnet(nn.Module):
    def __init__(self, batch_size, num_classes,
                 glove_path, vocab_size, 
                 input_dim, hidden_dim, 
                 embedding_dim = 300, 
                 dropout_rate = 0.1):
        super(rnet, self).__init__()
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Embedding
        self.glove_path = glove_path
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = GloveEmbedding(glove_path = self.glove_path, 
                                        vocab_size = self.vocab_size, 
                                        embedding_dim = self.embedding_dim)
        self.entity_embedding = EntityEmbedding(entity_path = self.entity_path,
                                                rel_size = self.rel_size,
                                                rel_dim = self.rel_dim)
        
        # LSTM - Encode
        self.encoder = RNNEncoder(self.batch_size, self.input_dim, self.hidden_dim, self.dropout_rate, 
                                  mode = 'LSTM', bidirectional = True)
        # mLSTM
        self.m_lstm_attention = mLstm(2 * self.hidden_dim, self.hidden_dim, self.batch_size, IsGate=True, MatchWeightFunction='Bi')
        
        # selfMatching
        self.self_matching = mLstm(self.hidden_dim, self.hidden_dim, self.batch_size, IsGate=True, MatchWeightFunction = 'Bi')
        
        # fully connect
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
                nn.Dropout(p = self.dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Dropout(p = self.dropout_rate),
                nn.Linear(self.hidden_dim, 1)
        )
        
    def forward(self, answers):
        
        answer_hiddens = []
        for answer in answers:
            # Glove Embedding and biLSTM Encoder
            text_embed, text_mask = self.embedding(answer[0])
            cor_embed, cor_mask = self.embedding(answer[1])
            text_encode = self.encoder(text_embed, text_mask)
            cor_encode = self.encoder(cor_embed, cor_mask)
            #print(text_encode.shape, cor_encode.shape)
            
            # GateAttention
            v_states = self.m_lstm_attention(cor_encode, text_encode) # batch * seq_len * hidden_dim
            v_states = torch.stack(v_states, 1)

            # selfMatching
            h_states = self.self_matching(v_states, v_states) #batch * seq_len * hidden_dim
            h_states = torch.stack(h_states, 1)
            h_states_max = torch.max(h_states, 1)[0] # batch * hidden_dim
            answer_hiddens.append(h_states_max)
    
        # Fully connect layer
        answer_stack = torch.stack(answer_hiddens, 1)
        predict_op = self.classifier(answer_stack).squeeze()
        return predict_op

class InfoHopNet(nn.Module):
    def __init__(self, 
                 bert_path,
                 bert_dim,
                 hidden_dim, 
                 hop_num = 3,
                 head_num = 4,
                 dropout_rate = 0.1):
        super(InfoHopNet, self).__init__()
        self.dropout_rate = dropout_rate
        
        # Bert
        #self.encoder = RNNEncoder(self.batch_size, self.input_dim, self.hidden_dim, self.dropout_rate, 
        #                                    mode = 'LSTM', bidirectional = True, output_need = 'hc')
        self.bert_path = bert_path
        self.bert_dim = bert_dim
        self.encoder = BertModel.from_pretrained(config.bert_root_dir + bert_path)
        
        # attention between answer and conceptnet information
        self.attention = InterAttention(self.bert_dim)

        # encode with knowledge
        self.hidden_dim = hidden_dim
        self.info_encoder = nn.GRUCell(self.bert_dim, self.hidden_dim)

        # multihop_attention
        self.hop_num = hop_num
        self.head_num = head_num
        self.passage_select = AttentionRank(self.hidden_dim, self.head_num, self.dropout_rate)

        # classfier
        self.classifier = nn.Sequential(
                nn.Linear(4 * self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Dropout(p = self.dropout_rate),
                nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, choices, is_training = False):
        # Input Shape:
        #  - choices: 4 * []
        #    - question_answer: batch * seq_len ([CLS]question[SEP]answer)
        #    - question_concept: batch * max_concept_num * seq_len ([CLS]question[SEP]concept)
        #    - question_es: batch * max_es_num * seq_len ([CLS]question[SEP]es)

        q_hiddens, a_hiddens = [], []
        for choice in choices: 
            # answer
            answer_seg, answer_mask = get_segment_ids(choice[0]), get_mask_ids(choice[0])
            answer_encode = self.encoder(choice[0], token_type_ids = answer_seg, attention_mask = answer_mask, output_all_encoded_layers = False)[0][:, 0, :]
            a_hiddens.append(answer_encode)
            #answer_encode: batch * bert_dim
            if not is_training:
                logger.info("answer shape: {}, answer_encode shape: {}".format(choice[0].shape, answer_encode.shape))

            # concept
            batch, sentence_num = choice[1].shape[0:2]
            concept = choice[1].view(batch * sentence_num, -1)
            concept_seg, concept_mask = get_segment_ids(concept), get_mask_ids(concept)
            concept_encode = self.encoder(concept, token_type_ids = concept_seg, attention_mask = concept_mask, output_all_encoded_layers = False)[0][:, 0, :]
            concept_encode = concept_encode.view(batch, sentence_num, self.hidden_dim)
            if not is_training:
                logger.info("concept shape: {}, concept_encode shape: {}".format(choice[1].shape, concept_encode.shape))

            # es passage
            batch, sentence_num = choice[2].shape[0:2]
            passage = choice[2].view(batch * sentence_num, -1)
            passage_seg, passage_mask = get_segment_ids(passage), get_mask_ids(passage)
            passage_encode = self.encoder(passage, token_type_ids = passage_seg, attention_mask = passage_mask, output_all_encoded_layers = False)[0][:, 0, :]
            passage_encode = passage_encode.view(batch, sentence_num, self.hidden_dim)
            if not is_training:
                logger.info("passage shape: {}, passage_encode shape: {}".format(choice[2].shape, passage_encode.shape))

            # conceptnet attention with option
            con_ans_encode = self.attention(answer_encode.unsqueeze(1), concept_encode)[0].squeeze(1)
            if not is_training:
                logger.info("con_ans_shape shape: {}".format(con_ans_encode.shape))
                logger.info("-----------------------Multihop-----------------------")
            
            h_states = [con_ans_encode]
            for hop_num in range(self.hop_num):
                select_sentence = self.passage_select(h_states[hop_num], passage_encode, is_training) # batch * hidden_dim
                h_states.append(self.info_encoder(select_sentence, h_states[hop_num]))
            h_states = torch.stack(h_states, 1) # batch * (hop_num+1) * hidden_dim
            if not is_training:
                logger.info("h_states shape: {}".format(h_states.shape))
            q_hiddens.append(h_states)

            
        # q_hiddens: 4 * [batch * (1+hop_num) * hidden_dim]  a_hiddens: 4 * [batch * hidden_dim]
        q_hiddens = torch.stack(q_hiddens, 1)
        a_hiddens = torch.stack(a_hiddens, 1)
        if not is_training:
            logger.info("q_hiddens shape: {}, a_hiddens shape: {}".format(q_hiddens.shape, a_hiddens.shape))

        # q_hiddens: batch * 4 * (hop_num+1) * hidden_dim
        # q_hiddens: batch * 4 * hidden_dim
        a_hiddens = a_hiddens.unsqueeze(2).repeat(1, 1, self.hop_num + 1, 1) # batch * 4 * hop_num * hidden_dim
        answer_passage_cat = torch.cat((a_hiddens, q_hiddens, a_hiddens - q_hiddens, a_hiddens * q_hiddens), -1) # batch * 4 * hop_num * (4 * hidden_dim)
        if not is_training:
            logger.info("answer_passage_cat shape: {}".format(answer_passage_cat.shape))

        output = self.classifier(answer_passage_cat).squeeze(-1) # batch * 4 * hop_num
        output = torch.max(output, 2)

        return output


class MemoryHopNet(nn.Module):
    def __init__(self, batch_size, hidden_dim, bert_parameter,
                 head_num = 4,
                 hop_num = 3,
                 dropout_rate = 0.1):
        super(MemoryHopNet, self).__init__()
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.hop_num = hop_num
        
        # LSTM - Encode // 可能要换成GRU
        self.bert_model_path = config.bert_root_dir + bert_parameter
        self.bert_encode = BertModel.from_pretrained(self.bert_model_path)
        #for param in self.bert_encode.parameters():
        #    param.requires_grad = False
        
        # multihop_attention
        self.multihop = MultihopAttention(self.batch_size, self.hidden_dim, self.head_num, self.hop_num, self.dropout_rate)

        # classfier
        self.classifier = nn.Sequential(
                nn.Linear(4 * self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Dropout(p = self.dropout_rate),
                nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, question, choices, texts, is_training = False):
        # question: batch * seq_len
        # answer: 4 * batch * seq_len
        # texts: 4 * batch * sentence_num * seq_len
        # print("question: ", question.shape)
        # print("chioces: ", choices.shape)
        # print("texts: ", texts.shape)
        probs, hiddens, answers_encode = [], [], []
        question_encode = self.bert_encode(question, output_all_encoded_layers=False)[0]# batch * seq_len * hidden_dim
        question_last_encode = question_encode[:, 0, :] # batch * hidden_dim
        #print("question_encode: ", question_last_encode.shape)

        for choice, text in zip(choices, texts):
            choice_encode = self.bert_encode(choice, output_all_encoded_layers=False)[0] # batch * seq_len * hidden_dim
            choice_last_encode = choice_encode[:, 0, :] # batch * hidden_dim
            answers_encode.append(choice_last_encode)
            #print("answer_encode: ", answer_encode.shape)

            text_batch, text_sentence = text.shape[0:2]
            text = text.view(text_batch * text_sentence, -1)
            text_encode = self.bert_encode(text, output_all_encoded_layers=False)[0] # batch * seq_len * hidden_dim
            text_encode = text_encode[:, 0, :].view(text_batch, text_sentence, -1) #  batch * hidden_dim
            
            #print("-----------------------Multihop-----------------------")
            prob, hidden = self.multihop(question_last_encode, text_encode, is_training)
            probs.append(prob)
            hiddens.append(hidden)
            
        # prob: batch * 4; hiddens: batch * 4 * hop_num * hidden_dim
        #print("-----------------------Answer-----------------------")
        prob, hiddens = torch.stack(probs, 1), torch.stack(hiddens, 1)
        #print("prob: ", prob.shape)
        #print("hiddens: ", hiddens.shape)
        
        # prob: batch * 4; 
        # the probability of four answers
        predict_prob = prob 

        # hiddens: batch * 4 * hop_num * hidden_dim
        # answer: batch * 4 * hidden_dim
        answers_encode = torch.stack(answers_encode, 1).unsqueeze(2).repeat(1, 1, self.hop_num, 1) # batch * 4 * hop_num * hidden_dim
        #print(answers_encode.shape)
        answer_hidden_cat = torch.cat((answers_encode, hiddens, answers_encode - hiddens, answers_encode * hiddens), -1) # batch * 4 * hop_num * (4 * hidden_dim)
        #print("answer cat hidden: ",answer_hidden_cat.shape)

        output = self.classifier(answer_hidden_cat).squeeze(-1) # batch * 4 * hop_num
        output = torch.max(output, 2)[0]

        return output
            



                
                
                








            


            
