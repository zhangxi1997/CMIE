import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'networks')
from q_v_transformer import CoAttention
from gcn import AdjLearner, GCN
from block import fusions #pytorch >= 1.1.0
import numpy as np
import random


class HGA(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, device):
        """
        Reasoning with Heterogeneous Graph Alignment for Video Question Answering (AAAI2020)
        :param vid_encoder:
        :param qns_encoder:
        :param device:
        """
        super(HGA, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.device = device
        hidden_size = vid_encoder.dim_hidden
        input_dropout_p = vid_encoder.input_dropout_p

        self.q_input_ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.v_input_ln = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.co_attn = CoAttention(
            hidden_size, n_layers=vid_encoder.n_layers, dropout_p=input_dropout_p)

        self.adj_learner = AdjLearner(
            hidden_size, hidden_size, dropout=input_dropout_p)

        self.gcn = GCN(
            hidden_size,
            hidden_size,
            hidden_size,
            num_layers=2,
            dropout=input_dropout_p)

        self.gcn_atten_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=-1)) #change to dim=-2 for attention-pooling otherwise sum-pooling

        self.global_fusion = fusions.Block(
            [hidden_size, hidden_size], hidden_size, dropout_input=input_dropout_p)

        self.fusion = fusions.Block([hidden_size, hidden_size], 1)

        self.linear_out = nn.Linear(hidden_size, 1)

        bert_dic_c = torch.tensor(np.load('dataset/nextqa/action_bert.npy'),
                                  dtype=torch.float32)
        self.prior_c = torch.tensor(np.load('dataset/nextqa/action_distribution_bert.npy'),
                                    dtype=torch.float)
        self.prior_c = self.prior_c.to(self.device)

        self.embedding_size = hidden_size
        self.Wy = nn.Linear(hidden_size, self.embedding_size)
        self.Wz = nn.Linear(768, self.embedding_size)
        self.W1 = nn.Linear(hidden_size, self.embedding_size)
        self.W2 = nn.Linear(768, self.embedding_size)
        self.W3 = nn.Linear(2 * self.embedding_size, self.embedding_size)

        # dic_dim = 768
        # dic_c = torch.Tensor(self.glove_dic_c.shape[0], dic_dim)
        self.dic_c_embedd = torch.nn.Parameter(bert_dic_c, requires_grad=True) # learnable
        self.dic_c_embedd.retain_grad() # important!!!



    def forward(self, vid_feats, qas, qas_lengths):
        """
        :param vid_feats:
        :param qns:
        :param qns_lengths:
        :param mode:
        :return:
        """
        if self.qns_encoder.use_bert:
            cand_qas = qas.permute(1, 0, 2, 3)  # for BERT
        else:
            cand_qas = qas.permute(1, 0, 2)

        cand_len = qas_lengths.permute(1, 0)

        app_feat = vid_feats[:,:,:2048]
        mot_feat = vid_feats[:,:,2048:]

        new_vid_feats = torch.cat([app_feat, mot_feat], dim=2)

        v_output, v_hidden = self.vid_encoder(new_vid_feats)
        v_last_hidden = torch.squeeze(v_hidden)

        out = []
        out_counterfactual = []

        for idx, qa in enumerate(cand_qas):
            encoder_out = self.vq_encoder(v_output, v_last_hidden, qa, cand_len[idx])
            out.append(encoder_out)
            encoder_out_counterfactual = self.vq_encoder_counterfactual(v_output, v_last_hidden, qa, cand_len[idx])
            out_counterfactual.append(encoder_out_counterfactual)

        out = torch.stack(out, 0).transpose(1, 0)
        out_counterfactual = torch.stack(out_counterfactual, 0).transpose(1, 0)
        _, predict_idx = torch.max(out, 1)
        _, predict_idx_counterfactual = torch.max(out_counterfactual, 1)

        return out, out_counterfactual, predict_idx, predict_idx_counterfactual


    def vq_encoder(self, v_output, v_last_hidden, qas, qas_lengths):
        """
        :param vid_feats:
        :param qas:
        :param qas_lengths:
        :return:
        """
        q_output, s_hidden = self.qns_encoder(qas, qas_lengths)
        qns_last_hidden = torch.squeeze(s_hidden)

        q_output = self.q_input_ln(q_output)
        v_output = self.v_input_ln(v_output)

        q_output, v_output, _, _ = self.co_attn(q_output, v_output)

        ### GCN
        adj = self.adj_learner(q_output, v_output)
        q_v_inputs = torch.cat((q_output, v_output), dim=1)
        q_v_output = self.gcn(q_v_inputs, adj)

        ## attention pool
        local_attn = self.gcn_atten_pool(q_v_output)
        local_out = torch.sum(q_v_output * local_attn, dim=1) # all ones

        # causal intervention
        qvc = self.z_dic(local_out, self.dic_c_embedd, self.prior_c)  # [bs, 2*256]
        qvc = self.W3(qvc)

        global_out = self.global_fusion((qns_last_hidden, v_last_hidden))
        out = self.fusion((global_out, qvc)).squeeze()

        return out

    def vq_encoder_counterfactual(self, v_output, v_last_hidden, qas, qas_lengths):
        """
        :param vid_feats:
        :param qas:
        :param qas_lengths:
        :return:
        """
        q_output, s_hidden = self.qns_encoder(qas, qas_lengths)
        qns_last_hidden = torch.squeeze(s_hidden)

        q_output = self.q_input_ln(q_output)
        v_output = self.v_input_ln(v_output)

        # counterfactual interaction
        _, _, q_output_counter, v_output_counter = self.co_attn(q_output, v_output)

        adj = self.adj_learner(q_output_counter, v_output_counter)
        q_v_inputs = torch.cat((q_output_counter, v_output_counter), dim=1)
        q_v_output = self.gcn(q_v_inputs, adj)

        ## attention pool
        local_attn = self.gcn_atten_pool(q_v_output)
        local_out = torch.sum(q_v_output * local_attn, dim=1)  # all ones

        # causal intervention
        qvc = self.z_dic(local_out, self.dic_c_embedd.squeeze(1), self.prior_c)
        qvc = self.W3(qvc)

        global_out = self.global_fusion((qns_last_hidden, v_last_hidden))
        out = self.fusion((global_out, qvc)).squeeze()

        return out

    def z_dic(self, qv, dic_c, prior_c):
        """
        Please note that we computer the intervention in the whole batch rather than for one object in the main paper.
        """
        # qv = [64,256]
        # dic_c = [992,300]
        # prior_c = [992]
        attention = torch.mm(self.Wy(qv), self.Wz(dic_c).t()) / (self.embedding_size ** 0.5)
        attention = torch.nn.functional.softmax(attention,1) # attention = [bs, 992]
        c_hat = attention.unsqueeze(2) * dic_c.unsqueeze(0) # c_hat = [bs, 992, 300]
        c = torch.matmul(prior_c.unsqueeze(0), c_hat).squeeze(1) # c = [bs,300]

        qvc = torch.cat([self.W1(qv), self.W2(c)], dim=-1)

        if torch.isnan(qvc).sum():
            print(qvc)
        return qvc
