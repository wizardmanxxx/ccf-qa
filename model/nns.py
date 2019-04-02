import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=False, dropout=0.2):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.reset_params()
        self.dropout = nn.Dropout(p=dropout)

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, 'weight_hh_l{}'.format(i)))
            nn.init.kaiming_normal_(getattr(self.rnn, 'weight_ih_l{}'.format(i)))
            nn.init.constant_(getattr(self.rnn, 'bias_hh_l{}'.format(i)), val=0)
            nn.init.constant_(getattr(self.rnn, 'bias_ih_l{}'.format(i)), val=0)
            getattr(self.rnn, 'bias_hh_l{}'.format(i)).chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, 'weight_hh_l{}_reverse'.format(i)))
                nn.init.kaiming_normal_(getattr(self.rnn, 'weight_ih_l{}_reverse'.format(i)))
                nn.init.constant_(getattr(self.rnn, 'bias_hh_l{}_reverse'.format(i)), val=0)
                nn.init.constant_(getattr(self.rnn, 'bias_ih_l{}_reverse'.format(i)), val=0)
                getattr(self.rnn, 'bias_hh_l{}_reverse'.format(i)).chunk(4)[1].fill_(1)

    def forward(self, x, x_len):
        x = self.dropout(x)

        x_len_sorted, x_idx = torch.sort(x_len, descending=True)
        x_sorted = x.index_select(dim=0, index=x_idx)
        _, x_ori_idx = torch.sort(x_idx)

        # pad_sentence = torch.sum(torch.eq(x_len, 0))
        # x_packed_cut = x_sorted[:-pad_sentence, :, :]
        # x_len_cut = x_len_sorted[:-pad_sentence].int()

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted.int(), batch_first=True)
        x_packed, (h, c) = self.rnn(x_packed)

        x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)[0]
        # tmp = torch.zeros([pad_sentence, x.size()[1], x.size()[2]])
        # x = torch.cat((x, tmp), 0)
        x = x.index_select(dim=0, index=x_ori_idx)
        # h = h.permute(1, 0, 2).contiguous().view(-1, h.size(0) * h.size(2)).squeeze()
        # h = h.index_select(dim=0, index=x_ori_idx)

        return x


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x


class BiMatchLSTM(torch.nn.Module):
    """
    inputs: input_p:    batch_size * max_passage_num x padded_p_len x hidden_size * 2
            mask_p:     batch_size * max_passage_num x padded_p_len
            input_q:    batch_size * max_passage_num x padded_q_len x hidden_size * 2
            mask_q:     batch_size * max_passage_num x padded_q_len
    outputs: encoding:   batch x time x hid
             last state: batch x hid
    """

    def __init__(self, input_p_dim, input_q_dim, nhids):
        super(BiMatchLSTM, self).__init__()
        self.nhids = nhids
        self.input_p_dim = input_p_dim
        self.input_q_dim = input_q_dim
        self.attention_layer = MatchLSTMAttention(input_p_dim, input_q_dim, output_dim=self.nhids)

        self.forward_rnn = MatchLSTM(input_p_dim=self.input_p_dim, input_q_dim=self.input_q_dim,
                                     nhids=self.nhids, attention_layer=self.attention_layer)

        self.backward_rnn = MatchLSTM(input_p_dim=self.input_p_dim, input_q_dim=self.input_q_dim,
                                      nhids=self.nhids, attention_layer=self.attention_layer)

    def flip(self, tensor, flip_dim=0):
        # flip
        idx = [i for i in range(tensor.size(flip_dim) - 1, -1, -1)]
        idx = torch.autograd.Variable(torch.LongTensor(idx))
        idx = idx.cuda()
        inverted_tensor = tensor.index_select(flip_dim, idx)
        return inverted_tensor

    def forward(self, input_p, mask_p, input_q, mask_q):
        # forward pass
        forward_states = self.forward_rnn.forward(input_p, mask_p, input_q, mask_q)
        forward_last_state = forward_states[:, -1]  # batch x hid

        # backward pass
        input_p_inverted = self.flip(input_p, flip_dim=1)  # batch x time x p_dim (backward)
        mask_p_inverted = self.flip(mask_p, flip_dim=1)  # batch x time (backward)
        backward_states = self.backward_rnn.forward(input_p_inverted, mask_p_inverted, input_q, mask_q)
        backward_last_state = backward_states[:, -1]  # batch x hid
        backward_states = self.flip(backward_states, flip_dim=1)  # batch x time x hid

        concat_states = torch.cat([forward_states, backward_states], -1)  # batch x time x hid * 2
        concat_states = concat_states * mask_p.unsqueeze(-1)  # batch x time x hid * 2
        concat_last_state = torch.cat([forward_last_state, backward_last_state], -1)  # batch x hid * 2

        return concat_states, concat_last_state


class MatchLSTM(torch.nn.Module):
    """
    inputs: p:          batch x time x inp_p
            mask_p:     batch x time
            q:          batch x time x inp_q
            mask_q:     batch x time
    outputs:
            encoding:   batch x time x h
            mask_p:     batch x time
    """

    def __init__(self, input_p_dim, input_q_dim, nhids, attention_layer):
        super(MatchLSTM, self).__init__()
        self.nhids = nhids
        self.input_p_dim = input_p_dim
        self.input_q_dim = input_q_dim
        self.attention_layer = attention_layer
        self.lstm_cell = torch.nn.LSTMCell(self.input_p_dim + self.input_q_dim, self.nhids)

    def get_init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(torch.autograd.Variable(weight.new(bsz, self.nhids).zero_()).cuda(),
                 torch.autograd.Variable(weight.new(bsz, self.nhids).zero_()).cuda())]

    def forward(self, input_p, mask_p, input_q, mask_q):
        batch_size = input_p.size(0)
        state_stp = self.get_init_hidden(batch_size)

        for t in range(input_p.size(1)):
            input_mask = mask_p[:, t]  # batch_size
            input_mask = input_mask.unsqueeze(1)  # batch_size x None
            curr_input = input_p[:, t]  # batch_size x inp_p
            previous_h, previous_c = state_stp[t]
            drop_input = self.attention_layer(curr_input, input_q, mask_q, h_tm1=previous_h)
            new_h, new_c = self.lstm_cell(drop_input, (previous_h, previous_c))
            new_h = new_h * input_mask + previous_h * (1 - input_mask)
            new_c = new_c * input_mask + previous_c * (1 - input_mask)
            state_stp.append((new_h, new_c))

        states = [h[0] for h in state_stp[1:]]  # list of batch x hid
        states = torch.stack(states, 1)  # batch x time x hid
        return states
