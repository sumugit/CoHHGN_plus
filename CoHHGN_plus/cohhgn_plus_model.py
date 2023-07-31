import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
from tqdm import tqdm
from cohhgn_plus_aggregator import GlobalAggregator
from config.config import Config
cf = Config()

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda(f'cuda:{cf.cuda}')
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class HyperConv(Module):
    def __init__(self, layers, dataset, emb_size, n_node, n_price, n_categoryBig, n_categoryMiddle, n_gender, n_region):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset
        self.n_node = n_node
        self.n_price = n_price
        self.n_categoryBig = n_categoryBig
        self.n_categoryMiddle = n_categoryMiddle
        self.n_gender = n_gender
        self.n_region = n_region

        self.mat_vp = nn.Parameter(torch.Tensor(self.n_node, 1))    # equ(4) attention vecor
        self.mat_vcb = nn.Parameter(torch.Tensor(self.n_node, 1)) # equ(4) attention vecor
        self.mat_vcm = nn.Parameter(torch.Tensor(self.n_node, 1)) # equ(4) attention vecor

        self.mat_pcb = nn.Parameter(torch.Tensor(self.n_price, 1))    # equ(4) attention vecor
        self.mat_pcm = nn.Parameter(torch.Tensor(self.n_price, 1))    # equ(4) attention vecor
        self.mat_pv = nn.Parameter(torch.Tensor(self.n_price, 1))    # equ(4) attention vecor
        self.mat_cbp = nn.Parameter(torch.Tensor(self.n_categoryBig, 1)) # equ(4) attention vecor
        self.mat_cbv = nn.Parameter(torch.Tensor(self.n_categoryBig, 1)) # equ(4) attention vecor
        self.mat_cbcm = nn.Parameter(torch.Tensor(self.n_categoryBig, 1)) # equ(4) attention vecor
        self.mat_cmp = nn.Parameter(torch.Tensor(self.n_categoryMiddle, 1)) # equ(4) attention vecor
        self.mat_cmv = nn.Parameter(torch.Tensor(self.n_categoryMiddle, 1)) # equ(4) attention vecor
        self.mat_cmcb = nn.Parameter(torch.Tensor(self.n_categoryMiddle, 1)) # equ(4) attention vecor

        self.b_o_gi_all = nn.Linear(self.emb_size, 1)
        self.b_o_gp_all = nn.Linear(self.emb_size, 1)
        self.b_o_gcb_all = nn.Linear(self.emb_size, 1)
        self.b_o_gcm_all = nn.Linear(self.emb_size, 1)
        
        self.a_o_g_i = nn.Linear(self.emb_size, self.emb_size)   # equ(6) 1st term item_emb
        self.b_o_gi1 = nn.Linear(self.emb_size, self.emb_size)       # equ(6) 2nd term item_emb
        self.b_o_gi2 = nn.Linear(self.emb_size, self.emb_size)       # equ(6) 3rd term item_emb
        self.b_o_gi3 = nn.Linear(self.emb_size, self.emb_size)       # equ(6) 3rd term item_emb

        self.a_o_g_p = nn.Linear(self.emb_size, self.emb_size)   # equ(6) 1st term price_emb
        self.b_o_gp1 = nn.Linear(self.emb_size, self.emb_size)       # equ(6) 2nd term price_emb
        self.b_o_gp2 = nn.Linear(self.emb_size, self.emb_size)       # equ(6) 3rd term price_emb
        self.b_o_gp3 = nn.Linear(self.emb_size, self.emb_size)       # equ(6) 3rd term price_emb

        self.a_o_g_cb = nn.Linear(self.emb_size, self.emb_size)   # equ(6) 1st term categoryBig_emb
        self.b_o_gcb1 = nn.Linear(self.emb_size, self.emb_size)       # equ(6) 2nd term categoryBig_emb
        self.b_o_gcb2 = nn.Linear(self.emb_size, self.emb_size)       # equ(6) 3rd term categoryBig_emb
        self.b_o_gcb3 = nn.Linear(self.emb_size, self.emb_size)       # equ(6) 3rd term categoryBig_emb
        
        self.a_o_g_cm = nn.Linear(self.emb_size, self.emb_size)   # equ(6) 1st term categoryMiddle_emb
        self.b_o_gcm1 = nn.Linear(self.emb_size, self.emb_size)       # equ(6) 2nd term categoryMiddle_emb
        self.b_o_gcm2 = nn.Linear(self.emb_size, self.emb_size)       # equ(6) 3rd term categoryMiddle_emb
        self.b_o_gcm3 = nn.Linear(self.emb_size, self.emb_size)       # equ(6) 3rd term categoryMiddle_emb

        self.dropout10 = nn.Dropout(0.1)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout60 = nn.Dropout(0.6)
        self.dropout70 = nn.Dropout(0.7)

    def forward(self, adjacency, adjacency_pv, adjacency_vp, adjacency_pcb, adjacency_cbp,
                adjacency_cbv, adjacency_vcb, adjacency_pcm, adjacency_cmp,
                adjacency_cmv, adjacency_vcm, adjacency_cbcm, adjacency_cmcb,
                item_emb, pri_emb, cateBig_emb, cateMiddle_emb):
        # get feature embedding using feature hyperedge
        for i in range(self.layers):
            # equ(9)
            item_embeddings = self.inter_gate(
                                        self.b_o_gi_all,
                                        self.b_o_gi_all,
                                        self.b_o_gi_all,
                                        self.b_o_gi_all,
                                        item_emb,
                                        self.intra_gate(adjacency_vp, self.mat_vp, pri_emb),
                                        self.intra_gate(adjacency_vcb, self.mat_vcb, cateBig_emb),
                                        self.intra_gate(adjacency_vcm, self.mat_vcm, cateMiddle_emb)                                                                              
                                        ) + self.get_embedding(adjacency, item_emb)
            # equ(10)
            price_embeddings = self.inter_gate(
                                        self.b_o_gp_all,
                                        self.b_o_gp_all,
                                        self.b_o_gp_all,
                                        self.b_o_gp_all,
                                        pri_emb,
                                        self.intra_gate(adjacency_pv, self.mat_pv, item_emb),
                                        self.intra_gate(adjacency_pcb, self.mat_pcb, cateBig_emb),
                                        self.intra_gate(adjacency_pcm, self.mat_pcm, cateMiddle_emb)                                        
                                        )
            # equ(11)
            categoryBig_embeddings =  self.inter_gate(
                                        self.b_o_gcb_all,
                                        self.b_o_gcb_all,
                                        self.b_o_gcb_all,
                                        self.b_o_gcb_all,
                                        cateBig_emb,
                                        self.intra_gate(adjacency_cbp, self.mat_cbp, pri_emb),
                                        self.intra_gate(adjacency_cbv, self.mat_cbv, item_emb),
                                        self.intra_gate(adjacency_cbcm, self.mat_cbcm, cateMiddle_emb)                                     
                                        )
            
            categoryMiddle_embeddings =  self.inter_gate(
                                        self.b_o_gcm_all,
                                        self.b_o_gcm_all,
                                        self.b_o_gcm_all,
                                        self.b_o_gcm_all,
                                        cateMiddle_emb,
                                        self.intra_gate(adjacency_cmp, self.mat_cmp, pri_emb),
                                        self.intra_gate(adjacency_cmv, self.mat_cmv, item_emb),
                                        self.intra_gate(adjacency_cmcb, self.mat_cmcb, cateBig_emb)                                    
                                        )
            
            item_emb = item_embeddings
            pri_emb = price_embeddings
            cateBig_emb = categoryBig_embeddings
            cateMiddle_emb = categoryMiddle_embeddings

        return item_emb, pri_emb

    def get_embedding(self, adjacency, embedding):
        """セッション内でのアイテム間の関連性を表す接続行列から, アイテムの埋め込みを更新する
        """
        # avg in equ(9)
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        embs = embedding
        # matrix multiplication
        item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), embs)
        return item_embeddings

    def intra_gate(self, adjacency, mat_v, embedding2):
        # embedding2: Attention 対象の embedding
        # equ(5)
        # attention to get embedding of type, and then gate to get final type embedding
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        # Pytorchのスパーステンソルに変換
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        
        matrix = trans_to_cuda(adjacency.to_dense())
        mat_v = mat_v.expand(mat_v.shape[0], self.emb_size) # parameter (u in equ(4))
        # equ(4)
        alpha = torch.mm(mat_v, torch.transpose(embedding2, 1, 0))
        alpha = torch.nn.Softmax(dim=1)(alpha)
        # 要素が0のattention-scoreを0にする
        alpha = alpha * matrix
        # 行毎に正規化
        sum_alpha_row = torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + 1e-8
        alpha = alpha / sum_alpha_row
        # equ(3)
        type_embs = torch.mm(alpha, embedding2)
        item_embeddings = type_embs
        return self.dropout70(item_embeddings)

    def inter_gate(self, a_o_g, b_o_g1, b_o_g_cateBig, b_o_g_cateMiddle, emb_mat1, emb_e1, emb_e2, emb_e3):
        def compute_normalized_att_scores(attention_functions, embeddings):
            exp_scores = [torch.exp(fn(emb)) for fn, emb in zip(attention_functions, embeddings)]
            sum_exp_scores = sum(exp_scores)
            normalized_scores = [score / sum_exp_scores for score in exp_scores]
            return normalized_scores

        attention_functions = [a_o_g, b_o_g1, b_o_g_cateBig, b_o_g_cateMiddle]
        embeddings = [emb_mat1, emb_e1, emb_e2, emb_e3]

        att_scores = compute_normalized_att_scores(attention_functions, embeddings)

        h_embeddings = sum(att_score * emb for att_score, emb in zip(att_scores, embeddings))

        # original
        # all_emb1 = torch.cat([emb_mat1, emb_e1, emb_e2], 1)
        # gate1 = torch.sigmoid(a_o_g(all_emb1) + b_o_g1(emb_e1) + b_o_g2(emb_e2))
        # h_embeddings = emb_mat1 + gate1 * emb_e1 + (1 - gate1) * emb_e2
        return self.dropout50(h_embeddings)

class CoHHGN_plus(Module):
    def __init__(self, adjacency, adjacency_pv, adjacency_vp,
                 adjacency_pcb, adjacency_cbp, adjacency_cbv, adjacency_vcb,
                 adjacency_pcm, adjacency_cmp, adjacency_cmv, adjacency_vcm,
                 adjacency_cbcm, adjacency_cmcb,
                 item_adj, pri_adj, item_num, pri_num,
                 n_node, n_price, n_categoryBig, n_categoryMiddle, n_gender, n_region,
                 lr, layers, l2, beta, dataset, hop, activate, dropout_gcn, dropout_global, n_sample,
                 num_heads=4, emb_size=100, batch_size=100,
                 ):
        super(CoHHGN_plus, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.n_price = n_price
        self.n_categoryBig = n_categoryBig
        self.n_categoryMiddle= n_categoryMiddle
        self.n_sale = 2
        self.n_gender = n_gender
        self.n_region = n_region
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.hop = hop
        self.activate = activate
        self.dropout_gcn = dropout_gcn
        self.dropout_global = dropout_global
        self.n_sample = n_sample

        self.adjacency = adjacency
        self.adjacency_pv = adjacency_pv
        self.adjacency_vp = adjacency_vp
        self.adjacency_pcb = adjacency_pcb
        self.adjacency_cbp = adjacency_cbp
        self.adjacency_cbv = adjacency_cbv
        self.adjacency_vcb = adjacency_vcb
        self.adjacency_pcm = adjacency_pcm
        self.adjacency_cmp = adjacency_cmp
        self.adjacency_cmv = adjacency_cmv
        self.adjacency_vcm = adjacency_vcm
        self.adjacency_cbcm = adjacency_cbcm
        self.adjacency_cmcb = adjacency_cmcb
        
        self.item_adj = trans_to_cuda(torch.Tensor(item_adj)).long()
        self.pri_adj = trans_to_cuda(torch.Tensor(pri_adj)).long()
        self.item_num = trans_to_cuda(torch.Tensor(item_num)).float()
        self.pri_num = trans_to_cuda(torch.Tensor(pri_num)).float()

        # Global Aggregator
        self.global_agg = []
        for i in range(hop):
            if activate == 'relu':
                agg = GlobalAggregator(self.emb_size, dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.emb_size, dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.price_embedding = nn.Embedding(self.n_price, self.emb_size)
        self.categoryBig_embedding = nn.Embedding(self.n_categoryBig, self.emb_size)
        self.categoryMiddle_embedding = nn.Embedding(self.n_categoryMiddle, self.emb_size)

        self.pos_embedding = nn.Embedding(2000, self.emb_size)
        self.HyperGraph = HyperConv(self.layers, dataset, self.emb_size, 
                                    self.n_node, self.n_price, self.n_categoryBig, self.n_categoryMiddle, self.n_gender, self.n_region)

        self.w_1 = nn.Linear(self.emb_size*2, self.emb_size) # equ(16) NN
        self.w_t = nn.Linear(self.n_gender + self.n_region, self.emb_size)
        self.w_s = nn.Linear(16 + self.n_sale, self.emb_size)
        self.w_2 = nn.Linear(self.emb_size, 1) # equ(18) NN
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)  # equ(18) 1st term
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False) # equ(18) 2nd term

        self.p_g1 = nn.Linear(self.emb_size, self.emb_size)
        self.p_g2 = nn.Linear(self.emb_size, self.emb_size)
        self.v_g1 = nn.Linear(self.emb_size, self.emb_size)
        self.v_g2 = nn.Linear(self.emb_size, self.emb_size)
        
        self.w_pri_cat = nn.Linear(self.emb_size*2, self.emb_size)
        self.w_h_cat = nn.Linear(self.emb_size*2, self.emb_size)

        # self_attention
        if emb_size % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (emb_size, num_heads))
        # parameters setting
        self.num_heads = num_heads  # 4
        self.attention_head_size = int(emb_size / num_heads)  # 16  the dimension of attention head
        self.all_head_size = int(self.num_heads * self.attention_head_size)
        # query, key, value
        self.query = nn.Linear(self.emb_size, self.emb_size )  # 128, 128
        self.key = nn.Linear(self.emb_size, self.emb_size )
        self.value = nn.Linear(self.emb_size, self.emb_size )

        # gate5 & gate6
        self.w_pi_1 = nn.Linear(self.emb_size, self.emb_size, bias=True) # equ(20) NN
        self.w_pi_2 = nn.Linear(self.emb_size, self.emb_size, bias=True) # equ(21) NN
        self.w_c_z = nn.Linear(self.emb_size, self.emb_size, bias=True)  # equ(22) 1st term
        self.u_j_z = nn.Linear(self.emb_size, self.emb_size, bias=True)  # equ(22) 2nd term
        self.w_c_r = nn.Linear(self.emb_size, self.emb_size, bias=True)  # equ(23) 1st term
        self.u_j_r = nn.Linear(self.emb_size, self.emb_size, bias=True)  # equ(23) 2nd term
        self.w_p = nn.Linear(self.emb_size, self.emb_size, bias=True)    # equ(24) 1st term
        self.u_p = nn.Linear(self.emb_size, self.emb_size, bias=True)    # equ(24) 2nd term
        self.w_i = nn.Linear(self.emb_size, self.emb_size, bias=True)    # equ(25) 1st term
        self.u_i = nn.Linear(self.emb_size, self.emb_size, bias=True)    # equ(25) 2nd term

        self.dropout = nn.Dropout(0.2)
        self.emb_dropout = nn.Dropout(0.25)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.7)

        self.loss_function = nn.CrossEntropyLoss() # equ(30)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        # initialize He method
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def get_period_encoding(self, period, d):
        """週情報を d 次元のベクトルにエンコードする関数
        Args:
        - month (int or torch.Tensor): 1 から 52 までの整数で表される週情報
        - d (int): dim of positional encoding
        Returns:
        - encoding (torch.Tensor): エンコードされた週情報
        """
        if isinstance(period, int):
            period = trans_to_cuda(torch.tensor(period))
        
        encoding = trans_to_cuda(torch.zeros(d))
        
        for i in range(d // 2):
            freq = 1 / (2000 ** (2 * i / d))
            encoding[2*i] = torch.sin(freq * period * 2 * trans_to_cuda(torch.tensor([np.pi])))
            encoding[2*i+1] = torch.cos(freq * period * 2 * trans_to_cuda(torch.tensor([np.pi])))
            
        return encoding

    def generate_sess_emb(self, item_embedding, price_embedding, item_emb_global, price_emb_global,
                          session_item, price_seqs, session_len, reversed_sess_item, mask, period, sale, gender, region):
        zeros = trans_to_cuda(torch.FloatTensor(1, self.emb_size).fill_(0))
        # zeros = torch.zeros(1, self.emb_size)  # for different GPU
        mask = mask.float().unsqueeze(-1)
        # equ(12)
        price_embedding = torch.cat([zeros, price_embedding], 0)
        get_pri = lambda i: price_embedding[price_seqs[i]]
        # (batch_size, max_n_price, emb_size)
        seq_pri = trans_to_cuda(torch.FloatTensor(self.batch_size, list(price_seqs.shape)[1], self.emb_size).fill_(0))
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size) # for different GPU
        for i in torch.arange(price_seqs.shape[0]):
            seq_pri[i] = get_pri(i)
        
        gate1 = torch.sigmoid(self.p_g1(seq_pri) + self.p_g2(price_emb_global))
        seq_pri =  gate1 * seq_pri + (1 - gate1) * price_emb_global
        # seq_pri = seq_pri + price_emb_global
        # seq_pri = torch.maximum(seq_pri, price_emb_global)
        # seq_pri = torch.cat([seq_pri, price_emb_global], -1)
        # seq_pri = self.w_pri_cat(seq_pri)
    
        # equ(13)
        # Create attention_mask for self-attention mechanism
        attention_mask = mask.permute(0,2,1).unsqueeze(1)  # [bs, 1, 1, seqlen]
        # Apply a large negative value to the padding positions in the attention_mask
        attention_mask = (1.0 - attention_mask) * -10000.0

        # Calculate query, key, and value for self-attention mechanism
        mixed_query_layer = self.query(seq_pri)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(seq_pri)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(seq_pri)  # [bs, seqlen, hid_size]

        # Calculate attention_head_size
        attention_head_size = int(self.emb_size / self.num_heads)
        # Transpose input tensors for multi-head attention mechanism
        query_layer = self.transpose_for_scores(mixed_query_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer, attention_head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        ### Self-attention (head) start ###
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(attention_head_size)  # [bs, 8, seqlen, seqlen]
        # add mask，set padding to -10000
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # equ(14)
        # [bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
        ### Self-attention (head) end ###
        # concat each attention head
        new_context_layer_shape = context_layer.size()[:-2] + (self.emb_size,)  # [bs, seqlen, 128]
        sa_result = context_layer.view(*new_context_layer_shape)
        # Create a tensor representing the positions of items in each session
        item_pos = trans_to_cuda(torch.tensor(range(1, seq_pri.size()[1] + 1)))
        item_pos = item_pos.unsqueeze(0).expand_as(price_seqs)
        
        # equ(15)
        # Multiply item_pos by the mask to filter out padding positions
        item_pos = item_pos * mask.squeeze(2)
        # Find the maximum position for each session and expand it to match the dimensions of item_pos
        item_last_num = torch.max(item_pos, 1)[0].unsqueeze(1).expand_as(item_pos)
        # Create a tensor indicating the last position of interest for each session
        last_pos_t = torch.where(item_pos - item_last_num >= 0, trans_to_cuda(torch.tensor([1.0])),
                                 trans_to_cuda(torch.tensor([0.0])))
        # Multiply last_pos_t by sa_result to extract the last position of interest for each session
        last_interest = last_pos_t.unsqueeze(2).expand_as(sa_result) * sa_result # (batch_size, max_n_price, emb_size)
        price_pre = torch.sum(last_interest, 1)
        
        # equ(16) h^{id}
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        # (batch_size, max_n_node, emb_size)
        seq_h = trans_to_cuda(torch.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0))
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        
        gate2 = torch.sigmoid(self.v_g1(seq_h) + self.v_g2(item_emb_global))
        seq_h =  gate2 * seq_h + (1 - gate2) * item_emb_global
        # seq_h = seq_h + item_emb_global
        # seq_h = torch.maximum(seq_h, item_emb_global)
        # seq_h = torch.cat([seq_h, item_emb_global], -1)
        # seq_h = self.w_h_cat(seq_h)
        
        # equ(18) \var{v*}
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        length = seq_h.shape[1]
        hs = hs.unsqueeze(-2).repeat(1, length, 1)
        # equ(16) pos_i
        pos_emb = self.pos_embedding.weight[:length]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)
        pos_period_emb = trans_to_cuda(torch.FloatTensor(self.batch_size, 16).fill_(0))
        for i in torch.arange(self.batch_size):
            pos_period_emb[i] = self.get_period_encoding(period[i], 16)
        # equ(16)
        nh1 = self.w_1(torch.cat([pos_emb, seq_h], -1))
        nh_att = self.w_t(torch.cat([gender.unsqueeze(dim=1), region.unsqueeze(dim=1)], -1))
        nh_sale = self.w_s(torch.cat([pos_period_emb.unsqueeze(dim=1), sale.unsqueeze(dim=1)], -1))
        nh = torch.tanh(nh1 + nh_att + nh_sale)
        # nh = torch.tanh(nh1)
        # equ(18)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = self.w_2(nh)
        # equ(17)
        beta = beta * mask
        interest_pre = torch.sum(beta * seq_h, 1)

        # Co-guided Learning
        m_c = torch.tanh(self.w_pi_1(price_pre * interest_pre)) # equ(20)
        m_j = torch.tanh(self.w_pi_2(price_pre + interest_pre)) # equ(21)
        r_i = torch.sigmoid(self.w_c_z(m_c) + self.u_j_z(m_j))  # equ(22)
        r_p = torch.sigmoid(self.w_c_r(m_c) + self.u_j_r(m_j))  # equ(23)
        m_p = torch.tanh(self.w_p(price_pre * r_p) + self.u_p((1 - r_p) * interest_pre)) # equ(24)
        m_i = torch.tanh(self.w_i(interest_pre * r_i) + self.u_i((1 - r_i) * price_pre)) # equ(25)
        # enriching the semantics of price and interest preferences
        p_pre = (price_pre + m_i )* m_p # equ(26)
        i_pre = (interest_pre + m_p) * m_i # equ(27)
        return i_pre, p_pre
    
    def transpose_for_scores(self, x, attention_head_size):
        """Function to transpose input tensors for multi-head attention mechanism
            x'shape = [bs, seqlen, hid_size]
        """
        # Calculate the new shape for input tensor x to be used in multi-head attention mechanism
        new_x_shape = x.size()[:-1] + (self.num_heads, attention_head_size)  # [bs, seqlen, 8, 16]
        # Reshape input tensor x to match the calculated dimensions
        x = x.view(*new_x_shape)
        # Permute dimensions of the reshaped tensor x for further processing
        return x.permute(0, 2, 1, 3)

    def sample(self, inputs, n_sample, adj, num):
        # 共起関係の item を取り出す
        # view(-1): flatten 2d-tensor to 1d-tensor (batch_size × man_n_node)
        # self.adj_all[inputs.view(-1)]: get list(same session item sorted by number of co-occurance) for each batch_session item
        # self.num[input.view(-1)]: get list(item-node, list(number of co-occurance of same session item)) for each batch_session item
        return adj[inputs.view(-1)], num[inputs.view(-1)] # (batch_size × max_n_node, n_sample)
    
    def global_emb(self, alias_inputs, session_item, reversed_sess_item, mask, embedding, adj, num):
        # Global Graph
        # batch 毎の session に含まれる商品(price)と共起関係にある商品(price)を抽出
        seqs_len = session_item.shape[1]
        item_neighbors = [session_item]
        weight_neighbors = []
        support_size = seqs_len # max_n_node (各 session に含まれる商品列)
        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.n_sample, adj, num)
            support_size *= self.n_sample
            item_neighbors.append(item_sample_i.view(self.batch_size, support_size)) # (batch_size, max_n_node × n_sample)
            weight_neighbors.append(weight_sample_i.view(self.batch_size, support_size)) # (batch_size, max_n_node × n_sample)
        # n-hop item embedding
        zeros = trans_to_cuda(torch.FloatTensor(1, self.emb_size).fill_(0))
        item_embedding = torch.cat([zeros, embedding.weight], 0)
        entity_vectors = [item_embedding[i] for i in item_neighbors]
        weight_vectors = weight_neighbors
        # compute equ(12) for each hop
        session_info = []
        # compute item embedding
        item_emb = item_embedding[reversed_sess_item] * mask.float().unsqueeze(-1) # (batch_size, max_n_node, hidden_size) * (batch_size, max_n_node, 1)
        # mean: equ(12)
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask.float(), -1).unsqueeze(-1)
        
        # sum
        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))
        
        # hop 数分の共起関係を学習
        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [self.batch_size, -1, self.n_sample, self.emb_size]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop+1].view(shape),
                                    masks=None,
                                    batch_size=self.batch_size,
                                    neighbor_weight=weight_vectors[hop].view(self.batch_size, -1, self.n_sample),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(self.batch_size, seqs_len, self.emb_size)
        # combine: equ(10)
        h_global = self.dropout7(h_global) # (batch_size, max_n_node, emb_size)
        
        # convert unique item index
        get = lambda index: h_global[index][alias_inputs[index]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()]) # (batch_size, max_n_node/max_n_price, emb_size)
        return seq_hidden

    def forward(self, item_alias_inputs, price_alias_inputs, session_item, price_seqs, session_len, reversed_sess_item, reversed_sess_price_seqs, mask, period, sale, gender, region):
        # get item and price preference: equ(26), equ(27)
        # session_item all sessions in a batch [[23,34,0,0],[1,3,4,0]]
        # (n_node, emb_size), (n_price, emb_size)
        item_embeddings_hg, price_embeddings_hg = self.HyperGraph(self.adjacency, self.adjacency_pv, self.adjacency_vp, 
                                                                  self.adjacency_pcb, self.adjacency_cbp, self.adjacency_cbv, self.adjacency_vcb,
                                                                  self.adjacency_pcm, self.adjacency_cmp, self.adjacency_cmv, self.adjacency_vcm,
                                                                  self.adjacency_cbcm, self.adjacency_cmcb,
                                                                  self.embedding.weight, self.price_embedding.weight,
                                                                  self.categoryBig_embedding.weight, self.categoryMiddle_embedding.weight,
                                                                  ) # updating the item embeddings
        # (batch_size, emb_size)
        item_emb_global = self.global_emb(item_alias_inputs, session_item, reversed_sess_item, mask, 
                                 self.embedding, self.item_adj, self.item_num)
        price_emb_global = self.global_emb(price_alias_inputs, price_seqs, reversed_sess_price_seqs, mask, 
                                 self.price_embedding, self.pri_adj, self.pri_num)
        # item_emb_global = None
        # price_emb_global = None

        # session embeddings in a batch (batch_size, emb_size)
        sess_emb_hgnn, sess_pri_hgnn = self.generate_sess_emb(item_embeddings_hg, price_embeddings_hg, item_emb_global, price_emb_global,
                                                              session_item, price_seqs, session_len, reversed_sess_item, mask, period, sale, gender, region)
        # get item-price table return price of items
        v_table = self.adjacency_vp.row
        temp, idx = torch.sort(torch.tensor(v_table), dim=0, descending=False)
        vp_idx = self.adjacency_vp.col[idx] # price col
        item_pri_l = price_embeddings_hg[vp_idx] # item info for price col (n_node, emb_size)
        return item_embeddings_hg, price_embeddings_hg, sess_emb_hgnn, sess_pri_hgnn, item_pri_l


def forward(model, i, data):
    """local forward fuction for each session
    Args:
        model (CoHHGN_plus): proposal model
        i (np.array): session batch index array
        data (tuple): processed data
    Returns:
        _type_: _description_
    """
    item_alias_inputs, price_alias_inputs, tar, session_len, session_item, reversed_sess_item, mask, price_seqs, reversed_price_seqs, period, sale, gender, region = data.get_slice(i) # obtaining instances from a batch
    # cuda option
    item_alias_inputs = trans_to_cuda(torch.Tensor(item_alias_inputs).long())
    price_alias_inputs = trans_to_cuda(torch.Tensor(price_alias_inputs).long())
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    price_seqs = trans_to_cuda(torch.Tensor(price_seqs).long())
    period = trans_to_cuda(torch.Tensor(period).float())
    sale = trans_to_cuda(torch.Tensor(sale).float())
    gender = trans_to_cuda(torch.Tensor(gender).float())
    region = trans_to_cuda(torch.Tensor(region).float())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    reversed_price_seqs = trans_to_cuda(torch.Tensor(reversed_price_seqs).long())
    # forward
    item_emb_hg, price_emb_hg, sess_emb_hgnn, sess_pri_hgnn, item_pri_l = model(item_alias_inputs, price_alias_inputs, session_item, price_seqs,
                                                                                session_len, reversed_sess_item, reversed_price_seqs,
                                                                                mask, period, sale, gender, region)
    # equ(28)
    scores_interest = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1, 0)) # equ(28) 2nd term
    scores_price = torch.mm(sess_pri_hgnn, torch.transpose(item_pri_l, 1, 0))     # equ(28) 1st term
    scores = scores_interest + scores_price
    return tar, scores


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    # train model CoHHGN_plus
    for i in tqdm(slices):
        # for each session batch
        model.zero_grad()
        # local forward function
        targets, scores = forward(model, i, train_data) # (batch_size, n_node)
        # equ(29), equ(30)
        loss = model.loss_function(scores + 1e-8, targets)
        loss = loss
        loss.backward()
        # print(loss.item())
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [1, 5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    # eval model CoHHGN_plus
    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in tqdm(slices):
        # for each session batch
        tar, scores = forward(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['ndcg%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['ndcg%d' % K].append(1 / (np.log2(np.where(prediction == target)[0][0] + 2)))
    return metrics, total_loss


def test(model, test_data, label_dict):
    # eval model CoHHGN_plus
    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    K = 20
    f = open('/workspace/sources/model/pred_log_20_35.txt', 'w')
    for i in tqdm(slices):
        # for each session batch
        tar, scores = forward(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        # i 追加
        for session, prediction, target in zip(i, index[:, :K], tar):
            f.write("######## session input ########\n")
            session_items = test_data.raw[session]
            session_items = [label_dict[str(item)] for item in session_items]
            for item in session_items:
                # print(item)
                f.write(item + '\n')
            f.write("######## target ########\n")
            f.write(label_dict[str(target+1)] + '\n')
            f.write("######## prediction ########\n")
            pred_items = [label_dict[str(pred+1)] for pred in prediction]
            for item in pred_items:   
                # print(item)
                f.write(item + '\n')
    f.close()