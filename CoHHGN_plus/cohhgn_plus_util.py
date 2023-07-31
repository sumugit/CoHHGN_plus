import numpy as np
# np.random.seed(42)
from scipy.sparse import csr_matrix
from operator import itemgetter
import torch

def data_masks(all_sessions, n_node):
    # same as preprocess
    # Heterogeneous Hypergraph の接続行列作成
    # csr_matrix (圧縮疎行列) の形式で扱う
    # indptr: 行の index, indices: 列の index, data: hyperedge に含まれれば 1
    # 行列のサイズ : (unique item1 num, unique item2 num)
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        # for each unique item1
        session = np.unique(all_sessions[j]) # unique item/priceLevel/categoryBig/categoryMiddle
        length = len(session)
        s = indptr[-1]
        # 1 行にユニークアイテム数分の要素と位置が記録される
        indptr.append((s + length))
        for i in range(length):
            # 列はアイテムの index
            indices.append(session[i]-1) # item_id - 1 が index
            # hyperedge (アイテムの種類数分ある) に含まれるので 1
            data.append(1)
    # indptr:sum of the session length; indices:item_id - 1
    # Compressed sparce matrix
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node)) # (unique item1 num, unique item2 num)
    return matrix

def data_easy_masks(data_l, n_row, n_col):
    # make csr_matrix
    data, indices, indptr  = data_l[0], data_l[1], data_l[2] # preprocess data_masks(tr_seqs)
    matrix = csr_matrix((data, indices, indptr), shape=(n_row, n_col))
    return matrix

def split_validation(train_set, valid_portion):
    # split data
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

# global graph
def handle_adj(adj_list, n_entity, sample_num, num_list=None):
    """sample co-occurance items from each item-node
    Args:
        adj_list (list): list(item-node, list(same session item sorted by number of co-occurance))
        n_entity (int): unique item count
        sample_num (int): sampling number
        num_list (list, optional): list(item-node, list(number of co-occurance of same session item)). Defaults to None.
    Returns: np.array, np.array
        adj_entity, num_entity: sampled adj_list, num_list
    """
    adj_entity = np.zeros([n_entity+1, sample_num], dtype=np.int64) # n_node + 1 にすること注意
    num_entity = np.zeros([n_entity+1, sample_num], dtype=np.int64)
    for entity in range(n_entity):
        # idx 0 is skipped
        neighbor = list(adj_list[entity])
        neighbor_weight = list(num_list[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        # row=0 はダミー (padding 用)
        adj_entity[entity+1] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity+1] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity

class Data():
    def __init__(self, data, shuffle=False,
                 n_node=None, n_price=None, n_categoryBig=None,
                 n_categoryMiddle=None, n_gender=None, n_region=None):
        self.raw = data[0]         # tr_seqs to np.array() (sessions, item_seq)
        self.price_raw = data[1]   # tr_pri to np.array()  (sessions, price_seq)
        # 各アイテム間のセッションに基づく関連性を表す正方行列 DHBH_T を作成
        H_T = data_easy_masks(data[2], len(data[0]), n_node) # csr_matrix (sessions, items)
        # 行毎に正規化
        BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1)) # csr_matrix (items, sessions)
        BH_T = BH_T.T # csr_matrix (sessions, items)
        ########################################
        H = H_T.T # csr_matrix (items, sessions)
        # 行毎に正規化
        DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1)) # csr_matrix (sessions, items)
        DH = DH.T # csr_matrix (items, sessions)
        DHBH_T = np.dot(DH, BH_T) # np.array (items, items)
        ########################################
        # それ以外の接続行列は attention-score の非ゼロ要素のみを抽出するのに用いる
        # priceLevel_item matrix
        H_pv = data_easy_masks(data[4], n_price, n_node)
        BH_pv = H_pv
        BH_vp = H_pv.T
        # priceLevel_categoryBig matrix
        H_pcb = data_easy_masks(data[5], n_price, n_categoryBig)
        BH_pcb = H_pcb
        BH_cbp = H_pcb.T
        # categoryBig_item matrix
        H_cbv = data_easy_masks(data[6], n_categoryBig, n_node)
        BH_cbv = H_cbv
        BH_vcb = H_cbv.T
        # priceLevel_categoryMiddle matrix
        H_pcm = data_easy_masks(data[7], n_price, n_categoryMiddle)
        BH_pcm = H_pcm
        BH_cmp = H_pcm.T
        # categoryMiddle_item matrix
        H_cmv = data_easy_masks(data[8], n_categoryMiddle, n_node)
        BH_cmv = H_cmv
        BH_vcm = H_cmv.T
        # categoryBig_categoryMiddle matrix
        H_cbcm = data_easy_masks(data[9], n_categoryBig, n_categoryMiddle)
        BH_cbcm = H_cbcm
        BH_cmcb = H_cbcm.T
        # adjacency matrix (セッション内でのアイテム間の関連性を表した行列)
        self.adjacency = DHBH_T.tocoo()
        # priceLevel-item
        self.adjacency_pv = BH_pv.tocoo()
        self.adjacency_vp = BH_vp.tocoo()
        # priceLevel-categoryBig
        self.adjacency_pcb = BH_pcb.tocoo()
        self.adjacency_cbp = BH_cbp.tocoo()
        # categoryBig-item
        self.adjacency_cbv = BH_cbv.tocoo()
        self.adjacency_vcb = BH_vcb.tocoo()
        # priceLevel-categoryBig
        self.adjacency_pcm = BH_pcm.tocoo()
        self.adjacency_cmp = BH_cmp.tocoo()
        # categoryBig-categoryMiddle
        self.adjacency_cbcm = BH_cbcm.tocoo()
        self.adjacency_cmcb = BH_cmcb.tocoo()
        # categoryMiddle-item
        self.adjacency_cmv = BH_cmv.tocoo()
        self.adjacency_vcm = BH_vcm.tocoo()
        
        self.n_node = n_node
        self.n_price = n_price
        self.n_categoryBig = n_categoryBig
        self.n_categoryMiddle = n_categoryMiddle
        self.n_sale = 2
        self.n_gender = n_gender
        self.n_region = n_region
        
        self.periods = np.asarray(data[10])
        self.super_sales = np.asarray(data[11])
        self.run_sales = np.asarray(data[12])
        self.genders = np.asarray(data[13])   # tr_gender/te_gender
        self.regions = np.asarray(data[14])   # tr_region/te_region
        self.targets = np.asarray(data[15])   # tr_labs/te_labs
        self.length = len(self.raw)           # session length
        self.shuffle = shuffle

    def get_overlap(self, sessions):
        # 未使用
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        """batch 毎の index を返す
        """
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            # random session item_seq, price_seq, target
            self.raw = [self.raw[idx] for idx in shuffled_arg] # self.raw[shuffled_arg]
            self.price_raw = [self.price_raw[idx] for idx in shuffled_arg] # self.price_raw[shuffled_arg]
            self.periods = self.periods[shuffled_arg]
            self.super_sales = self.super_sales[shuffled_arg]
            self.run_sales = self.run_sales[shuffled_arg]
            self.genders = self.genders[shuffled_arg]
            self.regions = self.regions[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
    
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        # padding (GCE-GNN と同じ)
        items, num_node, price_seqs = [], [], []
        inp = [self.raw[idx] for idx in index] # self.raw[index]
        inp_price = [self.price_raw[idx] for idx in index] # self.price_raw[index]
        inp_period = self.periods[index]
        inp_super_sale = self.super_sales[index]
        inp_run_sale = self.run_sales[index]
        inp_gender = self.genders[index]
        inp_region = self.regions[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        reversed_price_seqs = []
        mask = []
        sales = []
        periods = []
        genders = []
        regions = []
        for session, price, period, super_sale, run_sale, gender, region in zip(inp, inp_price, inp_period, inp_super_sale, inp_run_sale, inp_gender, inp_region):
            # for each session in a batch
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            price_seqs.append(price + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_price_seqs.append(list(reversed(price)) + (max_n_node - len(nonzero_elems)) * [0])            
            
            periods.append(period)
            sale_elem = np.zeros(self.n_sale, dtype=float)
            sale_elem[0] = super_sale - 1
            sale_elem[1] = run_sale - 1
            sales.append(sale_elem)
            gender_elem = np.zeros(self.n_gender, dtype=float)
            gender_elem[gender-1] = 1.0
            genders.append(gender_elem)
            region_elem = np.zeros(self.n_region, dtype=float)
            region_elem[region-1] = 1.0
            regions.append(region_elem)
        
        item_nodes = [np.unique(u_input) for u_input in items] # list
        item_alias_inputs = [[np.where(node == i)[0][0] for i in u_input] for u_input, node in zip(items, item_nodes)] # unique item index for each item in the session
        price_nodes = [np.unique(u_input) for u_input in price_seqs]
        price_alias_inputs = [[np.where(node == i)[0][0] for i in u_input] for u_input, node in zip(price_seqs, price_nodes)]

        return item_alias_inputs, price_alias_inputs, self.targets[index]-1, session_len, items, reversed_sess_item, mask, price_seqs, reversed_price_seqs, periods, sales, genders, regions
