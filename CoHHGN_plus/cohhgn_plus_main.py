'''
creat by yuta at August 2023
Reference: https://github.com/sumugit/CoHHGN_plus
'''

import sys
sys.path.append("/workspace/sources")
import argparse
import pickle
import time
from cohhgn_plus_util import Data, split_validation, handle_adj
from cohhgn_plus_model import *
import os
import random
from config.config import Config
cf = Config()
import warnings
import torch
import json
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
# Local Graph
parser.add_argument('--dataset', default=cf.age_range, help='dataset name: amazon/digineticaBuy/cosmetics/')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=128, help='embedding size')
parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads') # 4
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=3, help='the number of layer used, 2 for amazon, 3 for others')
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
parser.add_argument('--model', default=None, help='load pretrained model')
# Global Graph
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--hop', type=int, default=1)                                         # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0.6, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')    # [0, 0.5]
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')

opt = parser.parse_args()
print(opt)

torch.cuda.set_device(cf.cuda)

def torch_fix_seed(seed=cf.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def main():
    # list[0]:session, list[11]:label
    train_data = pickle.load(open('/workspace/datasets/object/' + opt.dataset + '/rk_train.txt', 'rb'))
    test_data = pickle.load(open('/workspace/datasets/object/' + opt.dataset + '/rk_test.txt', 'rb'))
    item_adj = pickle.load(open('/workspace/datasets/object/' + opt.dataset + '/item_adj_12.pkl', 'rb'))
    pri_adj = pickle.load(open('/workspace/datasets/object/' + opt.dataset + '/pri_adj_12.pkl', 'rb'))
    item_num = pickle.load(open('/workspace/datasets/object/' + opt.dataset + '/item_num_12.pkl', 'rb'))
    pri_num = pickle.load(open('/workspace/datasets/object/' + opt.dataset + '/pri_num_12.pkl', 'rb'))
    
    if opt.dataset == '20_35':
        n_node = 2344
        n_price = 10
        n_categoryBig = 36
        n_categoryMiddle = 325
        n_gender = 3
        n_region = 9
    elif opt.dataset == '35_50':
        n_node = 2570
        n_price = 10
        n_categoryBig = 36
        n_categoryMiddle = 332
        n_gender = 3
        n_region = 9
    elif opt.dataset == '50_65':
        n_node = 2306
        n_price = 10
        n_categoryBig = 36
        n_categoryMiddle = 320
        n_gender = 3
        n_region = 9
    elif opt.dataset == '65_80':
        n_node = 1820
        n_price = 10
        n_categoryBig = 35
        n_categoryMiddle = 292
        n_gender = 3
        n_region = 9
    else:
        print("unkonwn dataset")
        raise NotImplementedError
    
    # data_formate: sessions, price_seq, matrix_session_item, matrix_session_price, matrix_pv, matrix_pb, matrix_pc, matrix_bv, matrix_bc, matrix_cv
    train_data = Data(train_data, shuffle=True, 
                      n_node=n_node, n_price=n_price, n_categoryBig=n_categoryBig, 
                      n_categoryMiddle=n_categoryMiddle, n_gender=n_gender, n_region=n_region)
    test_data = Data(test_data, shuffle=False,
                     n_node=n_node, n_price=n_price, n_categoryBig=n_categoryBig,
                     n_categoryMiddle=n_categoryMiddle, n_gender=n_gender, n_region=n_region)
    
    item_adj, item_num = handle_adj(item_adj, n_node, opt.n_sample_all, item_num)  # sample co-occurance items from each item-node
    pri_adj, pri_num = handle_adj(pri_adj, n_price, opt.n_sample_all, pri_num)
    
    model = trans_to_cuda(CoHHGN_plus(
                        adjacency = train_data.adjacency,
                        adjacency_pv = train_data.adjacency_pv,
                        adjacency_vp = train_data.adjacency_vp,
                        adjacency_pcb = train_data.adjacency_pcb,
                        adjacency_cbp = train_data.adjacency_cbp,
                        adjacency_cbv = train_data.adjacency_cbv,
                        adjacency_vcb = train_data.adjacency_vcb,
                        adjacency_pcm = train_data.adjacency_pcm,
                        adjacency_cmp = train_data.adjacency_cmp,
                        adjacency_cmv = train_data.adjacency_cmv,
                        adjacency_vcm = train_data.adjacency_vcm,
                        adjacency_cbcm = train_data.adjacency_cbcm,
                        adjacency_cmcb = train_data.adjacency_cmcb,
                        n_node = n_node,
                        n_price = n_price,
                        n_categoryBig = n_categoryBig,
                        n_categoryMiddle = n_categoryMiddle,
                        n_gender = n_gender,
                        n_region = n_region,
                        item_adj = item_adj,
                        pri_adj = pri_adj,
                        item_num = item_num,
                        pri_num = pri_num,
                        lr = opt.lr,
                        l2 = opt.l2,
                        beta = opt.beta,
                        layers = opt.layer,
                        emb_size = opt.embSize,
                        batch_size = opt.batchSize,
                        dataset = opt.dataset,
                        num_heads = opt.num_heads,
                        hop = opt.hop,
                        activate = opt.activate,
                        dropout_gcn = opt.dropout_gcn,
                        dropout_global = opt.dropout_global,
                        n_sample = opt.n_sample))

    top_K = [1, 5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
                best_results['epoch%d' % K][2] = epoch
        print(metrics)
        print('P@1\tP@5\tM@5\tN@5\tP@10\tM@10\tN@10\tP@20\tM@20\tN@20\t')
        print("%.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f" % (
            best_results['metric1'][0], best_results['metric5'][0], best_results['metric5'][1],
            best_results['metric5'][2], best_results['metric10'][0], best_results['metric10'][1],
            best_results['metric10'][2], best_results['metric20'][0], best_results['metric20'][1],
            best_results['metric20'][2]))
        print("%d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d" % (
            best_results['epoch1'][0], best_results['epoch5'][0], best_results['epoch5'][1],
            best_results['epoch5'][2], best_results['epoch10'][0], best_results['epoch10'][1],
            best_results['epoch10'][2], best_results['epoch20'][0], best_results['epoch20'][1],
            best_results['epoch20'][2]))

if __name__ == '__main__':
    main()