import torch
import torch.nn as nn
import torch.nn.functional as F
from model.nns import LSTM, Linear
import numpy as np


# from utils.nn import LSTM, Linear


class BiDAF(nn.Module):
    def __init__(self, args, vocab):
        super(BiDAF, self).__init__()
        self.args = args
        self.vocab = vocab
        self.hidden_size = args.hidden_size
        self.word_embedding = nn.Embedding(self.vocab.size(), self.vocab.embed_dim)
        # initial word embedding
        self.word_embedding.weight.data.copy_(torch.from_numpy(self.vocab.embeddings))
        # self.word_embedding.weight.requires_grad = False
        self.q_encode = LSTM(self.vocab.embed_dim, self.hidden_size, bidirectional=True)
        self.p_encode = LSTM(self.vocab.embed_dim, self.hidden_size, bidirectional=True)
        self.dropout_prob = 0.1
        self.training = 1
        self.use_dropout = 1
        # 1. Character Embedding Layer
        # self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)
        # nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        #
        # self.char_conv = nn.Conv2d(1, args.char_channel_size, (args.char_dim, args.char_channel_width))

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        # self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

        # highway network
        # assert self.args.hidden_size * 2 == (self.args.char_channel_size + self.args.word_dim)
        # for i in range(2):
        #     setattr(self, 'highway_linear{}'.format(i),
        #             nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2),
        #                           nn.ReLU()))
        #     setattr(self, 'highway_gate{}'.format(i),
        #             nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2),
        #                           nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=args.hidden_size * 2,
                                 hidden_size=args.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=0)

        # 4. Attention Flow Layer
        self.att_weight_c = nn.Linear(args.hidden_size * 2, 1)
        self.att_weight_q = nn.Linear(args.hidden_size * 2, 1)
        self.att_weight_cq = nn.Linear(args.hidden_size * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = nn.LSTM(input_size=args.hidden_size * 8,
                                      hidden_size=args.hidden_size,
                                      bidirectional=True,
                                      batch_first=True,
                                      dropout=args.dropout)

        self.modeling_LSTM2 = nn.LSTM(input_size=args.hidden_size * 2,
                                      hidden_size=args.hidden_size,
                                      bidirectional=True,
                                      batch_first=True,
                                      dropout=args.dropout)

        # 6. Output Layer
        self.p1_weight_g = Linear(args.hidden_size * 8, 1, dropout=args.dropout)
        self.p1_weight_m = Linear(args.hidden_size * 2, 1, dropout=args.dropout)
        self.p2_weight_g = Linear(args.hidden_size * 8, 1, dropout=args.dropout)
        self.p2_weight_m = Linear(args.hidden_size * 2, 1, dropout=args.dropout)

        self.output_LSTM = nn.LSTM(input_size=args.hidden_size * 2,
                                   hidden_size=args.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=args.dropout)

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, p, q):
        # TODO: More memory-efficient architecture

        """
               input:
                  p: batch_size x padded_p_len
                  q: batch_size x padded_q_len
              output:
                  output: batch_size x padded_p_len x 2
              """

        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            cq = []
            for i in range(q_len):
                # (batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                # (batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze()
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, p_mask):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            p_mask = torch.eq(p_mask, 0)

            # (batch, c_len)
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM(m)[0]
            # (batch, c_len)
            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()
            p1 = F.softmax(p1.masked_fill(p_mask, -np.inf), dim=1)
            p2 = F.softmax(p2.masked_fill(p_mask, -np.inf), dim=1)

            return p1, p2

        # get mask
        q_mask = torch.ne(q, 0).float()  # batch_size x padded_q_len
        p_mask = torch.ne(p, 0).float()  # batch_size x padded_p_len
        q_lenth = torch.sum(q_mask, 1)
        p_lenth = torch.sum(p_mask, 1)
        # q = q.transpose(0, 1).contiguous()  # padded_q_len x batch_size
        q_emb = self.word_embedding(q)  # batch_size x padded_q_len x embed_dim
        q_output = self.q_encode(q_emb, q_lenth)
        # # batch_size x padded_q_len x hidden_size * 2
        # q_output = q_output.transpose(0, 1).contiguous()
        # batch_size x padded_q_len x hidden_size * 2
        # q_output = q_output * q_mask.unsqueeze(-1)
        # if self.use_dropout:
        #     q_output = torch.nn.functional.dropout(q_output, p=self.dropout_prob, training=self.training)

        # p = p.transpose(0, 1).contiguous()  # padded_p_len x batch_size
        p_emb = self.word_embedding(p)  # padded_p_len x batch_size x embed_dim
        p_output = self.p_encode(p_emb, p_lenth)

        # p_output = p_output.transpose(0, 1).contiguous()
        # # batch_size x padded_p_len x hidden_size * 2
        # p_output = p_output * p_mask.unsqueeze(-1)
        # if self.use_dropout:
        #     p_output = torch.nn.functional.dropout(p_output, p=self.dropout_prob, training=self.training)

        g = att_flow_layer(p_output, q_output)
        # 5. Modeling Layer
        m = self.modeling_LSTM2((self.modeling_LSTM1(g)[0]))[0]
        # 6. Output Layer
        p1, p2 = output_layer(g, m, p_mask)

        # (batch, c_len), (batch, c_len)
        return p1, p2
