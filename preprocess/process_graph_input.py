#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
from collections import defaultdict
import time
import csv
import pickle
import operator
import datetime
import json
import os
import sys
sys.path.append("/workspace/sources")
from config.config import Config
cf = Config()

print("-- Starting @ %ss" % datetime.datetime.now())

with open(cf.processed_path, "r") as f:
    reader = csv.DictReader(f, delimiter=',')
    sess_clicks = defaultdict(lambda: [])     # dict (key: session_id, value: item list)
    sess_date = {}                            # dict (key: session_id, value: date list) (used in train_test_split)
    ctr = 0                                   # counter
    curid = -1                                # current session_id
    curdate = None                            # current week
    for data in reader:
        sessid = data['session_id']
        # split date with session_id
        if curdate and curid != sessid:
            sess_date[curid] = int(data['week'])
        curid = sessid
        # item id
        item = data['genre_3']
        # current week
        curdate = data['week']
        sess_clicks[sessid].append(item)
        ctr += 1
    sess_date[curid] = int(data['week'])
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] > 9, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
dates = list(sess_date.items())
# Last one week for test
print('Splitting date to make train and test.')
split_date = 105
tra_sess = filter(lambda x: x[1] < split_date, dates)   # train session
tes_sess = filter(lambda x: x[1] >= split_date, dates)  # test session

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(unique session_id, week), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(unique session_id, week), (), ]
print(len(tra_sess))
print(len(tes_sess))
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtain_tra():
    train_ids = []   # session_id
    train_seqs = []  # item sequences by session_id
    train_dates = [] # week by session_id
    item_ctr = 1
    for session_id, date in tra_sess:
        seq = sess_clicks[session_id]
        outseq = []
        # 各session内で購入された商品をsession内出現順に変換
        # renumber items to start from 1
        for item in seq:
            # outseq.append(item)
            # if item not in item_dict:
            #     item_ctr += 1
            # item_dict[item] = True
            if item in item_dict:
                outseq.append(item_dict[item])
                # outseq.append(item)
            else:
                outseq.append(item_ctr)
                item_dict[item] = item_ctr
                item_ctr += 1
        # week の末尾で連続 session が切れる場合, 連続数が 2 未満になることがある
        if len(outseq) < 2:
            continue
        train_ids.append(session_id)
        train_dates.append(date)
        train_seqs.append(outseq)
    print(f"unique item count: {item_ctr}")
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtain_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for session_id, date in tes_sess:
        seq = sess_clicks[session_id]
        outseq = []
        for item in seq:
            # ignoring items that do not appear in training set
            if item in item_dict:
                outseq.append(item_dict[item])
                # outseq.append(item)
        if len(outseq) < 2:
            continue
        test_ids.append(session_id)
        test_dates.append(date)
        test_seqs.append(outseq)
    return test_ids, test_dates, test_seqs

# get train/test (session_ids, weeks, item sequences)
tra_ids, tra_dates, tra_seqs = obtain_tra()
tes_ids, tes_dates, tes_seqs = obtain_tes()


# 各 session 内 item をスライドさせている
# item の順序関係を逐次学習
# seq:[1, 2, 3]
# → labs (next item): [3, 2]
# → out_seqs (item substring): [[1, 2], [1]]
"""
def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq_date in enumerate(zip(iseqs, idates)):
        seq, date = seq_date
        for i in range(1, len(seq)):
            # i th item in id th session (id != session_id).
            tar = seq[-i]
            labs.append(tar)
            out_seqs.append(seq[:-i])
            # id th week
            out_dates.append(date)
            ids.append(id)
    return out_seqs, out_dates, labs, ids
"""

def process_seqs_no(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq_date in enumerate(zip(iseqs, idates)):
        seq, date = seq_date
        tar = seq[-1]
        labs.append(tar)
        out_seqs.append(seq[:-1])
        # id th week
        out_dates.append(date)
        ids.append(id)
    return out_seqs, out_dates, labs, ids


tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs_no(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs_no(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0
for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))
pickle.dump(tra, open(cf.srgnn_train_path, 'wb')) # (out_seqs, labs)
pickle.dump(tes, open(cf.srgnn_test_path, 'wb'))
pickle.dump(tra_seqs, open(cf.srgnn_all_train_seq_path, 'wb'))  # item sequences for each session_id
json_file = open(cf.srgnn_item_id_to_node_id_path, mode="w")
json.dump(item_dict, json_file, indent=2, ensure_ascii=False)
json_file.close()
print('Done.')
