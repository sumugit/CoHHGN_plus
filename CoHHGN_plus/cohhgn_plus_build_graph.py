"""build relation graph for global graph"""
import pickle
import argparse
import sys
sys.path.append("/workspace/sources")
from config.config import Config
cf = Config()
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=cf.age_range, help='20_35/35_50/50_65/65_80')
parser.add_argument('--sample_num', type=int, default=12)
opt = parser.parse_args()

dataset = opt.dataset
sample_num = opt.sample_num

# (session_num, item sequences)
item_seqs = pickle.load(open('/workspace/datasets/object/' + dataset + '/all_train_seq.txt', 'rb'))
pri_seqs = pickle.load(open('/workspace/datasets/object/' + dataset + '/all_train_price_seq.txt', 'rb'))

if dataset == '20_35':
    # unique item count
    num = 2344
elif dataset == "35_50":
    num = 2570
elif dataset == "50_65":
    num = 2306
elif dataset == "65_80":
    num = 1820
else:
    raise NotImplementedError()

def make_relation_graph(seq):
    relation = [] # consider co-occurance item
    neighbor = [] * num

    all_test = set()
    # list(item-node, dict(key:same session item, value: number of co-occurance))
    adj1 = [dict() for _ in range(num)]
    # list(item-node, list(same session item sorted by number of co-occurance))
    adj = [[] for _ in range(num)]
    # for each session
    for i in range(len(seq)):
        data = seq[i]
        # nearby 4 items (varepsilon = 4)
        for k in range(1, 4):
            for j in range(len(data)-k):
                relation.append([data[j]-1, data[j+k]-1]) # item_id - 1
                relation.append([data[j+k]-1, data[j]-1])
    # adj1
    for tup in relation:
        if tup[1] in adj1[tup[0]].keys():
            adj1[tup[0]][tup[1]] += 1
        else:
            adj1[tup[0]][tup[1]] = 1
    # list(item-node, list(number of co-occurance of same session item))
    weight = [[] for _ in range(num)]

    # for each item-node
    for i in range(num):
        # value (number of co-occurance) descent sort
        x = [v for v in sorted(adj1[i].items(), reverse=True, key=lambda x: x[1])]
        adj[i] = [v[0] for v in x] # same session item
        weight[i] = [v[1] for v in x] # number of co-occurance

    # filter same session item
    for i in range(num):
        adj[i] = adj[i][:sample_num]
        weight[i] = weight[i][:sample_num]

    return adj, weight

item_adj, item_weight = make_relation_graph(item_seqs)
pri_adj, pri_weight = make_relation_graph(pri_seqs)

# list(item-node, list(same session item sorted by number of co-occurance))
pickle.dump(item_adj, open(os.path.join(cf.OBJECT, 'item_adj_' + str(sample_num) + '.pkl'), 'wb'))
# list(item-node, list(number of co-occurance of same session item))
pickle.dump(item_weight, open(os.path.join(cf.OBJECT, 'item_num_' + str(sample_num) + '.pkl'), 'wb'))
# list(price-node, list(same session price sorted by number of co-occurance))
pickle.dump(pri_adj, open(os.path.join(cf.OBJECT, 'pri_adj_' + str(sample_num) + '.pkl'), 'wb'))
# list(price-node, list(number of co-occurance of same session price))
pickle.dump(pri_weight, open(os.path.join(cf.OBJECT, 'pri_num_' + str(sample_num) + '.pkl'), 'wb'))