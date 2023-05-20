
import os

def get_model(args):
    hidden_size = 100
    n_layers = 1
    emb_dim = 64
    model = NARM(args.product_num, hidden_size, emb_dim, n_layers, args).to(args.device)

    if not os.path.exists(os.path.join(args.output_root, args.model)):
        os.mkdir(os.path.join(args.output_root, args.model))

    return model



import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NARM(nn.Module):
    # 2017年ICKM
    # https://arxiv.org/pdf/1711.04725.pdf
    # 通过使用GRU处理序列化的会话，结合局部和全局编码操作，最终对用户行为进行预测。
    def __init__(self, n_items, hidden_size, embedding_dim, n_layers=1, args=None):
        super(NARM, self).__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(self.n_items + 1, self.embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.25) # 0.25
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers, batch_first=True)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(0.5) # 0.5
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        # self.sf = nn.Softmax()
        self.device = args.device

        # self.tanh = nn.Tanh()

    def forward(self, seq, lengths):
        hidden = self.init_hidden(seq.size(0))
        embs = self.emb_dropout(self.emb(seq))
        embs = pack_padded_sequence(embs, lengths, batch_first=True)
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out, batch_first=True)

        # fetch the last hidden state of last timestamp
        ht = hidden[-1]
        # gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())
        q2 = self.a_2(ht)
        mask = torch.where(seq > 0, torch.tensor([1.], device=self.device),
                           torch.tensor([0.], device=self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)

        # c_t = self.tanh(c_t)

        item_embs = self.emb(torch.arange(self.n_items + 1).to(self.device))
        scores = torch.matmul(c_t, self.b(item_embs).permute(1, 0))
        # scores = self.sf(scores)

        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)