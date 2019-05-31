import os
import codecs
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
 
def get_segment_ids(ids):
    # ids: batch_size * 4 * seq_len
    segment_matrix = torch.zeros(ids.shape)
    segment_idx = 102
    batch, choice = ids.shape[0:2]
        
    for b in range(batch):
        for c in range(choice):
            seg_idx = (ids[b][c] == 102).nonzero().squeeze()
            if seg_idx.dim() > 0 and len(seg_idx) > 0:
                segment_matrix[b][c][seg_idx[0]:] = 1
    return segment_matrix.type(torch.LongTensor).cuda()

def get_mask_ids(ids):
    return torch.sign(ids).cuda()

class GloveEmbedding(torch.nn.Module):
    def __init__(self, glove_path, vocab_size, embedding_dim):
        super(GloveEmbedding, self).__init__()
        self.glove_path = glove_path
        self.load_embedding_from_file = []
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        with codecs.open(glove_path, 'r', 'utf-8') as glove_f:
            for line in glove_f:
                word_vector = line.strip().split()
                vector = [float(num) for num in word_vector[1:]]
                self.load_embedding_from_file.append(vector)
        self.load_embedding_from_file = torch.tensor(self.load_embedding_from_file)
        assert self.load_embedding_from_file.dim() == 2, 'Embeddings parameter is expected to be 2-dimensional'
        self.embedding.weight = nn.Parameter(self.load_embedding_from_file, requires_grad = False)
        self.embedding.requires_grad = False    
        
    def forward(self, context):
        context_emb = self.embedding(context)
        context_mask = torch.sign(context)
        return context_emb, context_mask

class EntityEmbedding(torch.nn.Module):
    def __init__(self, entity_path, rel_size, vocab_size, rel_dim):
        super(EntityEmbedding, self).__init__()
        self.entity_path = entity_path
        self.rel_size = rel_size
        self.rel_dim = rel_dim
        self.vocab_size = vocab_size
        
        entity_i, entity_t = torch.load(self.entity_path)
        self.entity = torch.sparse.FloatTensor(entity_i, entity_t).to_dense()
        self.embedding = nn.Embedding(rel_size, rel_dim)
        self.embedding.weight = nn.Parameter(self.entity, requires_grad = False)
        self.embedding.requires_grad = False 
    
    def forward(self, context):
        entity_emb = self.embedding(context)
        return entity_emb

class RNNEncoder(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, dropout_rate, 
                     mode = 'LSTM', output_need = 'output', bidirectional = True):
        super(RNNEncoder, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.output_need = output_need
        self.bidirectional = bidirectional
        
        assert mode == 'LSTM' or mode == 'GRU'
        if mode == 'LSTM':
            self.enc_rnn = nn.LSTM(
                self.input_dim, self.hidden_dim, 1,
                bidirectional = self.bidirectional, 
                dropout=self.dropout_rate,
                batch_first = True)
        elif mode == 'GRU':
            self.enc_rnn = nn.GRU(
                self.input_dim, self.hidden_dim, 1,
                bidirectional = self.bidirectional, 
                dropout=self.dropout_rate,
                batch_first = True)
        else:
            raise NotImplementedError
            
    
    def forward(self, inputs, inputs_mask):
        inputs_mask = torch.sum(inputs_mask, -1)
        inputs_mask, idx_sort = torch.sort(inputs_mask, descending=True)
        inputs = inputs.index_select(0, idx_sort)
        
        inputs_packed = nn.utils.rnn.pack_padded_sequence(inputs, inputs_mask, batch_first = True)
        output, hc = self.enc_rnn(inputs_packed)
        if self.output_need == 'output':
            output = nn.utils.rnn.pad_packed_sequence(output, batch_first = True)[0] # batch_size * seq_len * (2*hidden_units)
            idx_unsort = torch.argsort(idx_sort)
            output = output.index_select(0, idx_unsort)
            return output
        elif self.output_need == 'hc':
            idx_unsort = torch.argsort(idx_sort)
            hc_0 = hc[0].index_select(1, idx_unsort)
            hc_1 = hc[1].index_select(1, idx_unsort)
            return (hc_0, hc_1)


class TriMatchAttention(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, entity_embedding, entity_lambda = [0, 0, 0]):
        super(TriMatchAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        
        self.W_s = nn.Linear(input_dim, hidden_dim)
        self.W_t = nn.Linear(input_dim, hidden_dim)
        self.W_r = nn.Linear(hidden_dim, hidden_dim)
        self.W_e = nn.Linear(hidden_dim, 1)

        self.entity_embedding = entity_embedding
        
    def forward(self, hs, ht_k, hm_k, hs_i = None, ht_k_i = None): 
        # hs: batch_size * seq_len * input_dim
        # ht_k: batch_size * 1 * input_dim
        # hm_k: batch_size * hidden_dim
        # hs_i: batch * seq_len
        # ht_k_i: batch * 1
        W_hs = self.W_s(hs) # batch * seq_len * hidden_dim
        W_ht = self.W_t(ht_k).unsqueeze(1) # batch * hidden_dim
        W_hm = self.W_r(hm_k).unsqueeze(1) # batch * 1 * hidden_dim
        #print("add_componet: ",W_hs.shape, W_ht.shape, W_hm.shape)
        linear_sum = W_hs + W_ht + W_hm
        #print("linear_sum: ", linear_sum.shape)
        W_sum = self.W_e(torch.tanh(linear_sum)).squeeze(2) # batch * seq_len

        if self.entity_embedding != None:
            self.entity_lambda = torch.tensor(entity_lambda).float().cuda()
            #print(hs_i, ht_k_i)
            rel_pos = hs_i * self.entity_embedding.vocab_size + ht_k_i.unsqueeze(1)
            rel_value = self.entity_embedding(rel_pos) # batch * seq_len * rel_dim
            rel_value = torch.matmul(rel_value, self.entity_lambda) # batch * seq_len
            W_sum = W_sum + rel_value
        else:
            rel_value = torch.zeros(hs_i.shape)  # batch * seq_len
        
        W_softmax = F.softmax(W_sum, 1).unsqueeze(1) # batch * 1 * seq_len 
        alpha = torch.bmm(W_softmax, W_hs).squeeze(1) # batch * input_dim
        #alpha =  weighted_sum(W_softmax, hs) # batch * input_dim
        #print("alpha: ", alpha.shape)
        return alpha

class BiMatchAttention(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiMatchAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        
        self.W_s = nn.Linear(input_dim, hidden_dim)
        self.W_t = nn.Linear(input_dim, hidden_dim)
        self.W_e = nn.Linear(hidden_dim, 1)
        
    def forward(self, hs, ht_k): 
        # hs: batch_size * seq_len * input_dim
        # ht_k: batch_size * 1 * input_dim
        W_hs = self.W_s(hs) # batch * seq_len * hidden_dim
        W_ht = self.W_t(ht_k).unsqueeze(1) # batch * hidden_dim
        #print("add_componet: ",W_hs.shape, W_ht.shape)
        linear_sum = W_hs + W_ht 
        #print("linear_sum: ", linear_sum.shape)
        W_sum = self.W_e(torch.tanh(linear_sum)).squeeze(2) # batch * seq_len
        W_softmax = F.softmax(W_sum, 1).unsqueeze(1) # batch * 1 * seq_len 
        alpha = torch.bmm(W_softmax, W_hs).squeeze() # batch * hidden_dim
        #print("alpha: ", alpha.shape)
        return alpha

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(MultiheadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.project_weight = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.project_dropout = nn.Dropout(p=self.dropout_rate)

        self.W_s = nn.Linear(hidden_dim, hidden_dim)
        self.W_t = nn.Linear(hidden_dim, hidden_dim)
        self.W_e = nn.Linear(hidden_dim, 1)
        
    def forward(self, question, text):
        # question: batch * hidden_dim
        # text: batch * sentence_num * hidden_dim
        text = self.project_weight(text)
        text = self.project_dropout(text)

        W_hs = self.W_s(text) # batch * sentence_num * hidden_dim
        W_ht = self.W_t(question).unsqueeze(1) # batch * 1 * hidden_dim
        #print(W_hs.shape, W_ht.shape)
        linear_sum = W_hs + W_ht 
        #print("linear_sum: ", linear_sum.shape)

        W_sum = self.W_e(torch.tanh(linear_sum)).squeeze(2) # batch * sentence_num
        W_softmax = F.softmax(W_sum, 1) # batch * sentence_num

        return W_softmax

class InterAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(InterAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.G = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.p_softmax = nn.Softmax(dim=-2)
        self.q_softmax = nn.Softmax(dim=-1)

    def forward(self, p_encode, q_encode):
        # p_encode: batch * p_len * hidden_dim
        # q_encode: batch * q_len * hidden_dim
        q_project = torch.transpose(self.G(q_encode), -2, -1)
        #print("q_project: ", q_project.shape) # batch * hidden_dim * q_len

        att_matrix = torch.bmm(p_encode, q_project) # batch * p_len * q_len
        #print("attention_matrix: ", att_matrix.shape)

        m_p = torch.bmm(self.q_softmax(att_matrix), q_encode) # batch * p_len * hidden_dim
        m_q = torch.bmm(torch.transpose(self.p_softmax(att_matrix), -2, -1) , p_encode) # batch * q_len * hidden_dim
        #print("m_p shape: {}, m_q shape: {}".format(m_p.shape, m_q.shape))

        return m_p, m_q


class mLstm(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, 
                 IsGate = False, MatchWeightFunction = 'Tri', 
                 entity_embedding = None, entity_lambda = [0, 0, 0]):
        super(mLstm, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.entity_embedding = entity_embedding
        self.entity_lambda = entity_lambda

        self.m_lstm_encoder = nn.LSTMCell(self.input_dim + self.hidden_dim, self.hidden_dim)
        if MatchWeightFunction == 'Tri':
            self.MatchWeightFunction = 'Tri'
            self.match_attention = TriMatchAttention(input_dim, hidden_dim, self.entity_embedding, self.entity_lambda)
        elif MatchWeightFunction == 'Bi':
            self.MatchWeightFunction = 'Bi'
            self.match_attention = BiMatchAttention(input_dim, hidden_dim)
        else:
            raise NotImplementedError

        if IsGate:
            self.IsGate = True
            self.W_g = nn.Linear(input_dim + hidden_dim, input_dim + hidden_dim)
        else:
            self.IsGate = False
        
    def forward(self, p, q, p_seq, q_seq):
        # p, q: batch * seq_len * hidden_dim
        # p_seq, q_seq: batch * seq_len
        p_len = p.shape[1]
        batch_size = p.shape[0]

        h_0 = p.new_zeros(batch_size, self.hidden_dim)
        c_states = [h_0]
        h_states = [h_0]
        
        for pos in range(p_len):
            p_pos = p[:, pos, :] # batch * input_dim
            p_seq_pos = p_seq[:, pos]
            hidden_pos = (h_states[pos], c_states[pos])
            if self.MatchWeightFunction == 'Tri':
                alpha = self.match_attention(q, p_pos, hidden_pos[0], q_seq, p_seq_pos) # batch * hidden_dim
            else:
                alpha = self.match_attention(q, p_pos) # batch * hidden_dim
            concat_alpha_htk = torch.cat((alpha, p_pos), 1) # batch * ( input_dim +  hidden_dim)
            if self.IsGate:
                g_t = torch.sigmoid(self.W_g(concat_alpha_htk))
                concat_alpha_htk = g_t * concat_alpha_htk
            hidden_cur = self.m_lstm_encoder(concat_alpha_htk, hidden_pos)
            h_states.append(hidden_cur[0])
            c_states.append(hidden_cur[1])
        return h_states[1:]

class AttentionRank(nn.Module):
    def __init__(self, hidden_dim, head_num = 4, dropout_rate = 0.1):
        super(AttentionRank, self).__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout_rate = dropout_rate

        # multihead attention
        self.multihead_attention = self.clones(MultiheadAttention(self.hidden_dim, self.dropout_rate), self.head_num)
        self.head_weight = nn.Linear(self.head_num, 1)

    def clones(self, module, N):
        import copy
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, question, passage, is_training):
        alpha = []
        
        for i in range(self.head_num):
            alpha_i = self.multihead_attention[i](question, passage) # batch * es_sentence_num
            alpha.append(alpha_i)

        alpha = torch.stack(alpha, 2) # batch * es_sentence_num * head_num 
        if not is_training:
            print("alpha shape: ", alpha.shape)

        alpha_linear = self.head_weight(alpha).squeeze(2) # batch * es_sentence_num
        max_alpha = torch.max(alpha_linear, 1)
        sentence_max_index = max_alpha[1] # batch
        
        max_sentence = []
        for i in range(question.shape[0]):
            max_text_encode = torch.index_select(passage[i], 0, sentence_max_index[i]).squeeze() # hidden_dim
            max_sentence.append(max_text_encode)
        max_sentence = torch.stack(max_sentence, 0)
        if not is_training:
            print("select sentence shape: ", max_sentence.shape)
        return max_sentence # batch * hidden_dim

class MultihopAttention(nn.Module):
    def __init__(self, batch_size, hidden_dim, head_num, hop_num, dropout_rate):
        super(MultihopAttention, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.hop_num = hop_num
        self.head_num = head_num
        self.dropout_rate = dropout_rate

        self.multihead_attention = self.clones(MultiheadAttention(self.hidden_dim, self.dropout_rate), self.head_num)
        self.head_weight = nn.Linear(self.head_num, 1)

        self.HopGRU = nn.GRUCell(self.hidden_dim * 2, self.hidden_dim)
        self.GRU_dropout = nn.Dropout(p=self.dropout_rate)
        self.GRU_layernorm = nn.LayerNorm(self.hidden_dim)
        
    
    def clones(self, module, N):
        import copy
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, question, text, is_training):
        # question: batch * hidden_dim
        # text: batch * sentence_num * text
        temp_batch_size = text.shape[0]
        h_0 = question.new_zeros(temp_batch_size, self.hidden_dim)
        h_states = [h_0]

        sentence_max_value = []
        for hop in range(self.hop_num):
            
            alpha = []
            #print("text: ", text.shape)
            for i in range(self.head_num):
                alpha_i = self.multihead_attention[i](question, text) # batch * text_sentence_num
                #print("alpha_i: ", alpha_i.shape)
                alpha.append(alpha_i)
            alpha = torch.stack(alpha, 2) # batch * test_sentence_num * head_num 
            #print("alpha: ", alpha.shape)

            alpha_linear = self.head_weight(alpha).squeeze(2) # batch * test_sentence_num

            scale_rate = torch.sqrt(torch.pow(alpha_linear, 2).sum(1, keepdim=True))
            prob = alpha_linear / scale_rate

            #print("alpha_linear: ", alpha_linear.shape)
            max_alpha = torch.max(alpha_linear, 1)
            sentence_max_index = max_alpha[1] # batch
            sentence_max_value.append(prob) # batch * sentence_num
            #print("alpha_index: ", max_alpha)

            if is_training:
                print(max_alpha)

            cat_q_t = []
            for i in range(temp_batch_size):
                max_text_encode = torch.index_select(text[i], 0, sentence_max_index[i]).squeeze() # hidden_dim
                cat_question_text = torch.cat((question[i], max_text_encode)) # hidden_dim * 2
                cat_q_t.append(cat_question_text)
            cat_q_t = torch.stack(cat_q_t, 0) # batch * (hidden_dim * 2)
            #print("question text concat: ", cat_q_t.shape)

            current_hidden = self.HopGRU(cat_q_t, h_states[hop])
            h_states.append(current_hidden)
            #print("hop hidden: ", current_hidden.shape)

            output = self.GRU_dropout(current_hidden)
            #output = self.GRU_layernorm(output)
            question = output
        
        predict_prob = sentence_max_value[0]
        h_states = torch.stack(h_states[1:], 1)

        #print("\npredict probabilty: ", predict_prob)
        #print("return h_states: ", h_states.shape)
        return predict_prob, h_states



                    
