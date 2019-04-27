import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import decoders
import encoders
from cuda import CUDA


class DeleteModel(nn.Module):
    def __init__(self, vocab_size, pad_id, config=None):
        super(DeleteModel, self).__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.config = config
        self.batch_size = config['data']['batch_size']
        self.options = config['model']
        
        self.embedding = nn.Embedding(self.vocab_size, self.options['emb_dim'], self.pad_id)
        self.attribute_embedding = nn.Embedding(num_embeddings=2, embedding_dim=self.options['emb_dim'])
        self.attr_size = self.options['emb_dim']
        
        if self.options['encoder'] == 'lstm':
            self.encoder = encoders.LSTMEncoder(self.options['emb_dim'], self.options['enc_hidden_dim'],
                                                self.options['enc_layers'], self.options['bidirectional'],
                                                self.options['dropout'])
            self.ctx_bridge = nn.Linear(self.options['enc_hidden_dim'], self.options['dec_hidden_dim'])
        else:
            # TODO: GRU encoder
            raise NotImplementedError('unknown encoder type')
        
        self.c_bridge = nn.Linear(self.attr_size + self.options['enc_hidden_dim'], self.options['dec_hidden_dim'])
        self.h_bridge = nn.Linear(self.attr_size + self.options['enc_hidden_dim'], self.options['dec_hidden_dim'])
        
        self.decoder = decoders.StackedAttentionLSTM(config=config)
        
        self.output_projection = nn.Linear(self.options['dec_hidden_dim'], self.vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.h_bridge.bias.data.fill_(0)
        self.c_bridge.bias.data.fill_(0)
        self.output_projection.bias.data.fill_(0)
        
    def forward(self, input_con, con_mask, con_len, input_attr, attr_mask, attr_len, input_data, mode):
        con_emb = self.embedding(input_con)
        con_mask = (1-con_mask).byte()
        output_con, (con_h_t, con_c_t) = self.encoder(con_emb, con_len, con_mask)
        
        if self.options['bidirectional']:
            # [batch, hidden_dim]
            h_t = torch.cat((con_h_t[-1], con_h_t[-2]), 1)
            c_t = torch.cat((con_c_t[-1], con_c_t[-2]), 1)
        else:
            # [batch, hidden_dim]
            h_t = con_h_t[-1]
            c_t = con_c_t[-1]
            
        output_con = self.ctx_bridge(output_con)
        
        # encode attribute info
        # [2, word_dim]
        a_ht = self.attribute_embedding(input_attr)
        # [batch, hidden_dim + attr_size]
        c_t = torch.cat((c_t, a_ht), -1)
        h_t = torch.cat((h_t, a_ht), -1)
        
        # [batch, hidden_dim]
        c_t = self.c_bridge(c_t)
        h_t = self.h_bridge(h_t)
        
        data_emb = self.embedding(input_data)
        # [batch, max_len, hidden_dim]
        output_data, (_, _) = self.decoder(data_emb, (h_t, c_t), output_con, con_mask)
        # [batch * max_len, hidden_dim]
        output_data_reshape = output_data.contiguous().view(output_data.size()[0]*output_data.size()[1], 
                                                     output_data.size()[2])
        # [batch * max_len, vocab_size]
        decoder_logit = self.output_projection(output_data_reshape)
        # [batch, max_len, vocab_size]
        decoder_logit = decoder_logit.view(output_data.size()[0], output_data.size()[1], 
                                           decoder_logit.size()[1])
        # [batch, max_len, vocab_size]
        probs = self.softmax(decoder_logit)
        
        return decoder_logit, probs
    
    
    def count_params(self):
        n_params = 0
        for param in self.parameters():
            n_params += np.prod(param.data.cpu().numpy().shape)
        return n_params
        

class DeleteRetrieveModel(nn.Module):
    def __init__(self, vocab_size, pad_id, config=None):
        super(DeleteRetrieveModel, self).__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.config = config
        self.batch_size = config['data']['batch_size']
        self.options = config['model']
        
        self.embedding = nn.Embedding(self.vocab_size, self.options['emb_dim'], self.pad_id)
        
        if self.options['encoder'] == 'lstm':
            self.encoder = encoders.LSTMEncoder(self.options['emb_dim'], self.options['enc_hidden_dim'],
                                                self.options['enc_layers'], self.options['bidirectional'],
                                                self.options['dropout'])
            self.ctx_bridge = nn.Linear(self.options['enc_hidden_dim'], self.options['dec_hidden_dim'])
        else:
            # TODO: GRU encoder
            raise NotImplementedError('unknown encoder type')
            
        self.attribute_encoder = encoders.LSTMEncoder(self.options['emb_dim'], self.options['enc_hidden_dim'],
                                                      self.options['enc_layers'], self.options['bidirectional'],
                                                      self.options['dropout'], pack=False)
        self.attr_size = self.options['enc_hidden_dim']
        
        self.c_bridge = nn.Linear(self.attr_size + self.options['enc_hidden_dim'], self.options['dec_hidden_dim'])
        self.h_bridge = nn.Linear(self.attr_size + self.options['enc_hidden_dim'], self.options['dec_hidden_dim'])
        
        self.decoder = decoders.StackedAttentionLSTM(config=config)
        
        self.output_projection = nn.Linear(self.options['dec_hidden_dim'], self.vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.h_bridge.bias.data.fill_(0)
        self.c_bridge.bias.data.fill_(0)
        self.output_projection.bias.data.fill_(0)
        
    def forward(self, input_con, con_mask, con_len, input_attr, attr_mask, attr_len, input_data, mode):
        # [batch, max_len, word_dim]
        con_emb = self.embedding(input_con)
        con_mask = (1-con_mask).byte()
        output_con, (con_h_t, con_c_t) = self.encoder(con_emb, con_len, con_mask)
        
        if self.options['bidirectional']:
            # [batch, hidden_dim]
            h_t = torch.cat((con_h_t[-1], con_h_t[-2]), 1)
            c_t = torch.cat((con_c_t[-1], con_c_t[-2]), 1)
        else:
            # [batch, hidden_dim]
            h_t = con_h_t[-1]
            c_t = con_c_t[-1]
            
        output_con = self.ctx_bridge(output_con)
        
        # encode attribute info
        attr_emb = self.embedding(input_attr)
        _, (a_ht, a_ct) = self.attribute_encoder(attr_emb, attr_len, attr_mask)
        if self.options['bidirectional']:
            # [batch, hidden_dim]
            a_ht = torch.cat((a_ht[-1], a_ht[-2]), 1)
            a_ct = torch.cat((a_ct[-1], a_ct[-2]), 1)
        else:
            # [batch, hidden_dim]
            a_ht = a_ht[-1]
            a_ct = a_ct[-1]
        # [batch, hidde_dim*2]
        h_t = torch.cat((h_t, a_ht), -1)
        c_t = torch.cat((c_t, a_ct), -1)

        # [batch, hidden_dim]
        c_t = self.c_bridge(c_t)
        h_t = self.h_bridge(h_t)
        
        
        data_emb = self.embedding(input_data)
        # [batch, max_len, hidden_dim]
        output_data, (_, _) = self.decoder(data_emb, (h_t, c_t), output_con, con_mask)
        # [batch * max_len, hidden_dim]
        output_data_reshape = output_data.contiguous().view(output_data.size()[0]*output_data.size()[1], 
                                                     output_data.size()[2])
        # [batch * max_len, vocab_size]
        decoder_logit = self.output_projection(output_data_reshape)
        # [batch, max_len, vocab_size]
        decoder_logit = decoder_logit.view(output_data.size()[0], output_data.size()[1], 
                                           decoder_logit.size()[1])
        # [batch, max_len, vocab_size]
        probs = self.softmax(decoder_logit)
        print(decoder_logit.size())
        print(probs.size())
        
        return decoder_logit, probs
    
    
    def count_params(self):
        n_params = 0
        for param in self.parameters():
            n_params += np.prod(param.data.cpu().numpy().shape)
        return n_params


class PointerModel(nn.Module):
    def __init__(self, vocab_size, pad_id, config=None):
        super(PointerModel, self).__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.config = config
        self.batch_size = config['data']['batch_size']
        self.options = config['model']
        
        self.embedding = nn.Embedding(self.vocab_size, self.options['emb_dim'], self.pad_id)
        
        if self.options['encoder'] == 'lstm':
            self.encoder = encoders.LSTMEncoder(self.options['emb_dim'], self.options['enc_hidden_dim'],
                                                self.options['enc_layers'], self.options['bidirectional'],
                                                self.options['dropout'])
            self.ctx_bridge = nn.Linear(self.options['enc_hidden_dim'], self.options['dec_hidden_dim'])
        else:
            # TODO: GRU encoder
            raise NotImplementedError('unknown encoder type')
            
        self.attribute_encoder = encoders.LSTMEncoder(self.options['emb_dim'], self.options['enc_hidden_dim'],
                                                      self.options['enc_layers'], self.options['bidirectional'],
                                                      self.options['dropout'], pack=False)
        self.attr_size = self.options['enc_hidden_dim']
        
        
        self.p_gen_linear = nn.Linear(self.options['enc_hidden_dim'] * 4, 1)
        
        
        self.c_bridge = nn.Linear(self.attr_size + self.options['enc_hidden_dim'], self.options['dec_hidden_dim'])
        self.h_bridge = nn.Linear(self.attr_size + self.options['enc_hidden_dim'], self.options['dec_hidden_dim'])
        
        self.decoder = decoders.StackedAttentionLSTM(config=config)
        
        self.output_projection = nn.Linear(self.options['dec_hidden_dim'], self.vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.h_bridge.bias.data.fill_(0)
        self.c_bridge.bias.data.fill_(0)
        self.output_projection.bias.data.fill_(0)
        
    def forward(self, input_con, con_mask, con_len, input_attr, attr_mask, attr_len, input_data, mode):
        # [batch, max_len, word_dim]
        con_emb = self.embedding(input_con)
        con_mask = (1-con_mask).byte()
        output_con, (con_h_t, con_c_t) = self.encoder(con_emb, con_len, con_mask)
        
        if self.options['bidirectional']:
            # [batch, hidden_dim]
            h_t = torch.cat((con_h_t[-1], con_h_t[-2]), 1)
            c_t = torch.cat((con_c_t[-1], con_c_t[-2]), 1)
        else:
            # [batch, hidden_dim]
            h_t = con_h_t[-1]
            c_t = con_c_t[-1]
            
        # [batch, hidden_dim]
        output_con = self.ctx_bridge(output_con)
        
        # encode attribute info
        attr_emb = self.embedding(input_attr)
        _, (a_ht, a_ct) = self.attribute_encoder(attr_emb, attr_len, attr_mask)
        if self.options['bidirectional']:
            # [batch, hidden_dim]
            a_ht = torch.cat((a_ht[-1], a_ht[-2]), 1)
            a_ct = torch.cat((a_ct[-1], a_ct[-2]), 1)
        else:
            # [batch, hidden_dim]
            a_ht = a_ht[-1]
            a_ct = a_ct[-1]
        # [batch, hidde_dim*2]
        h_t = torch.cat((h_t, a_ht), -1)
        c_t = torch.cat((c_t, a_ct), -1)

        # [batch, hidden_dim]
        c_t = self.c_bridge(c_t)
        h_t = self.h_bridge(h_t)
        
        if mode == 'train':
            decoder_logit_list = []
            final_dist_list = []
            for di in range(min(self.config['data']['max_len'], input_data.size()[1])):
                y_t = input_data[:, di]
                y_data_emb = self.embedding(y_t)
                y_data_emb = y_data_emb.unsqueeze(dim=1)
                # [batch, 1, hidden_dim]
                output_data, (h_t, c_t) = self.decoder(y_data_emb, (h_t, c_t), output_con, con_mask)
                
                h_t = h_t.squeeze()
                c_t = c_t.squeeze()
                
                dec_dist = torch.cat((h_t, c_t), 1)
                attr_dist = torch.cat((a_ht, a_ct), 1)
                p_gen_input = torch.cat((dec_dist, attr_dist), 1)
                p_gen = self.p_gen_linear(p_gen_input)
                p_gen = torch.sigmoid(p_gen)
            
                # [batch, hidden_dim]
                output_data_reshape = output_data.contiguous().view(output_data.size()[0]*output_data.size()[1], 
                                                         output_data.size()[2])
                # [batch, vocab_size]
                decoder_logit = self.output_projection(output_data_reshape)
                # [batch, vocab_size]
                dec_probs = self.softmax(decoder_logit)
                
                # [batch, vocab_size]
                attr_logit = self.output_projection(a_ht)
                # [batch, vocab_size]
                attr_probs = self.softmax(attr_logit)
                
                dec_probs = p_gen * dec_probs
                attr_probs = (1-p_gen) * attr_probs
                final_dist = dec_probs + attr_probs
                
                decoder_logit_list.append(decoder_logit)
                final_dist_list.append(final_dist)
            
            decoder_logits = torch.stack(decoder_logit_list, 1)
            final_dists = torch.stack(final_dist_list, 1)
        else:
            y_data_emb = self.embedding(input_data)
            # [batch, hidden_dim]
            output_data, (h_t, c_t) = self.decoder(y_data_emb, (h_t, c_t), output_con, con_mask)
            
            h_t = h_t.squeeze(dim=0)
            c_t = c_t.squeeze(dim=0)
            dec_dist = torch.cat((h_t, c_t), 1)
            attr_dist = torch.cat((a_ht, a_ct), 1)
            p_gen_input = torch.cat((dec_dist, attr_dist), 1)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)
        
            # [batch, vocab_size]
            decoder_logits = self.output_projection(output_data)
            # [batch, vocab_size]
            dec_probs = self.softmax(decoder_logits)
            
            # [batch, vocab_size]
            attr_logit = self.output_projection(a_ht)
            # [batch, vocab_size]
            attr_probs = self.softmax(attr_logit)
            
            dec_probs = p_gen * dec_probs
            attr_probs = (1-p_gen) * attr_probs
            final_dists = dec_probs + attr_probs
        
        return decoder_logits, final_dists
    
    
    def count_params(self):
        n_params = 0
        for param in self.parameters():
            n_params += np.prod(param.data.cpu().numpy().shape)
        return n_params
        


class GreedySearchDecoder(nn.Module):
    def __init__(self, model):
        super(GreedySearchDecoder, self).__init__()
        self.model = model
        
        
    def forward(self, input_con, con_mask, con_len, input_attr, attr_mask, attr_len, max_len, start_id):
        input_data = Variable(torch.LongTensor([[start_id] for i in range(input_con.size(0))]))
        if CUDA:
            input_data = input_data.cuda()
        
        for i in range(max_len):
            decoder_logit, word_prob = self.model(input_con, con_mask, con_len, 
                                             input_attr, attr_mask, attr_len, input_data, mode='dev')
            decoder_argmax = word_prob.data.cpu().numpy().argmax(axis=-1)
            next_pred = Variable(torch.from_numpy(decoder_argmax[:, -1]))
            if CUDA:
                next_pred = next_pred.cuda()
            input_data = torch.cat((input_data, next_pred.unsqueeze(1)), dim=1)
        
        return decoder_logit, input_data
