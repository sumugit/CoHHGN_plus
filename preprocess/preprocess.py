import os
import pickle
import time
import pandas as pd
import numpy as np
import math
import json
import sys
sys.path.append("/workspace/sources")
from config.config import Config
cf = Config()

# the number of price levels (\rho in equ(1))
price_level_num = 9
data_all = pd.read_csv(cf.processed_path)
data_all = data_all[['session_id', 'week', 'period', 'super_sale', 'run_sale', 'user_gender', 'user_region', 'price', 'genre_1', 'genre_2', 'genre_3']]


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

# check numeric formatting
def reg_price(price):
    if is_number(price):
        if price == 0:
            results = ''
        else:
            results = float(price)
    else:
        results = ''
    return results

# check label encoding
def reg_category(cate):
    results = ''
    if is_number(cate):
        results = cate
    else:
        results = ''
    return results

# cast
data_all['price'] = data_all.price.map(reg_price)
data_all['super_sale'] = data_all.super_sale.map(reg_category)
data_all['run_sale'] = data_all.run_sale.map(reg_category)
data_all['user_gender'] = data_all.user_gender.map(reg_category)
data_all['user_region'] = data_all.user_region.map(reg_category)
data_all['genre_1'] = data_all.genre_1.map(reg_category)
data_all['genre_2'] = data_all.genre_2.map(reg_category)

interaction = data_all[['session_id', 'week', 'period', 'super_sale', 'run_sale', 'user_gender', 'user_region', 'genre_3']]
item_all = data_all[['genre_3', 'genre_2', 'genre_1', 'price']]
item_all[['price']] = item_all[['price']].astype(float)

# make unique item dataFrame to get statistics information
item_all.drop_duplicates(subset=['genre_3'], keep='first', inplace=True)

# the number of big_category
group_big_cate_num = pd.DataFrame(item_all.groupby(item_all['genre_1']).count())
group_big_num = group_big_cate_num.reset_index()[['genre_1', 'genre_3']].rename(columns={'genre_3': 'cnt_genre_1'})
# the number of middle_category
group_middle_cate_num = pd.DataFrame(item_all.groupby(item_all['genre_2']).count())
group_middle_num = group_middle_cate_num.reset_index()[['genre_2', 'genre_3']].rename(columns={'genre_3': 'cnt_genre_2'})
#  min price in each category
group_cate_min = pd.DataFrame(item_all['price'].groupby(item_all['genre_1']).min())
group_min = group_cate_min.reset_index()[['genre_1', 'price']].rename(columns={'price': 'min'})
# max price in each category
group_cate_max = pd.DataFrame(item_all['price'].groupby(item_all['genre_1']).max())
group_max = group_cate_max.reset_index()[['genre_1', 'price']].rename(columns={'price': 'max'})
# average price in each category
group_cate_mean = pd.DataFrame(item_all['price'].groupby(item_all['genre_1']).mean())
group_mean = group_cate_mean.reset_index()[['genre_1', 'price']].rename(columns={'price': 'mean'})

group_cate_std = pd.DataFrame(item_all['price'].groupby(item_all['genre_1']).std())
group_std = group_cate_std.reset_index()[['genre_1', 'price']].rename(columns={'price': 'std'})

item_data1 = pd.merge(item_all, group_big_num, how='left', on='genre_1')
item_data2 = pd.merge(item_data1, group_middle_num, how='left', on='genre_2')
item_data3 = pd.merge(item_data2, group_min, how='left', on='genre_1')
item_data4 = pd.merge(item_data3, group_max, how='left', on='genre_1')
item_data5 = pd.merge(item_data4, group_mean, how='left', on='genre_1')
item_data = pd.merge(item_data5, group_std, how='left', on='genre_1')


def logistic(t, u, s):
    # equ(2)
    gama = s * 3 ** (0.5) / math.pi
    results = 1 / (1 + math.exp(-1*(t - u) / gama))
    return results

def get_price_level(price, p_min, p_max, mean, std):
    # equ(1)
    if std == 0:
        print('only one sample')
        return -1
    fenzi = logistic(price, mean, std) - logistic(p_min, mean, std)
    fenmu = logistic(p_max, mean, std) - logistic(p_min, mean, std)
    # nan
    if fenmu == 0 or price == 0:
        return -1
    results = int(fenzi / fenmu * price_level_num) + 1
    return results

# price normalization
item_data['price_level'] = item_data.apply(
    lambda row: get_price_level(row['price'], row['min'], row['max'], row['mean'], row['std']), axis=1)
# filter price is nan
item_final = item_data[item_data['price_level'] != -1]
# get price processed data
user_item1 = pd.merge(interaction, item_final, how='left', on='genre_3')
user_item2 = user_item1.dropna(axis=0)
# sort records
user_item2.sort_values(by=["session_id", "week"], inplace=True, ascending=[True, True])

user_click_num = pd.DataFrame(user_item2.groupby(user_item2['session_id']).count())
click_num = user_click_num.reset_index()[['session_id', 'genre_3']].rename(columns={'genre_3': 'click_num'})
item_data6 = pd.merge(user_item2, click_num, how='left', on='session_id')
data = item_data6[['session_id', 'week', 'period', 'super_sale', 'run_sale', 'user_gender', 'user_region', 'genre_3', 'genre_2', 'genre_1', 'price', 'price_level']]
# rename column
data_all = data.rename(
    columns={'session_id': 'sessionID',
             'genre_3': 'itemID',
             'genre_2': 'categoryMiddle',
             'genre_1': 'categoryBig',
             'week': 'time',
             'user_gender': 'gender',
             'user_region': 'region',
             'price_level': 'priceLevel',
             })
data_all = data_all[['sessionID', 'time', 'period', 'super_sale', 'run_sale', 'gender', 'region', 'itemID', 'categoryMiddle', 'categoryBig', 'price', 'priceLevel']]

# get graph input (train, test)
reviewerID2sessionID = {}
asin2itemID = {}
categoryBig2categoryBigID = {}
categoryMiddle2categoryMiddleID = {}
sessionNum = 0
itemNum = 0
categoryBigNum = 0
categoryMiddleNum = 0
# count session, item catgegory
for _, row in data_all.iterrows():
    if row['sessionID'] not in reviewerID2sessionID:
        sessionNum += 1
        reviewerID2sessionID[row['sessionID']] = sessionNum
    if row['itemID'] not in asin2itemID:
        itemNum += 1
        asin2itemID[row['itemID']] = itemNum
    if row['categoryBig'] not in categoryBig2categoryBigID:
        categoryBigNum += 1
        categoryBig2categoryBigID[row['categoryBig']] = categoryBigNum
    if row['categoryMiddle'] not in categoryMiddle2categoryMiddleID:
        categoryMiddleNum += 1
        categoryMiddle2categoryMiddleID[row['categoryMiddle']] = categoryMiddleNum
print('#session: ', sessionNum)
print('#item: ', itemNum)
print('#categoryBig: ', categoryBigNum)
print('#categoryMiddle: ', categoryMiddleNum)
# save dict
# reviewerID2sessionID
json_file = open(os.path.join(cf.JSON2, 'reviewerID2sessionID.json'), mode="w")
json.dump(reviewerID2sessionID, json_file, indent=2, ensure_ascii=False)
json_file.close()
# asin2itemID
json_file = open(os.path.join(cf.JSON2, 'asin2itemID.json'), mode="w")
json.dump(asin2itemID, json_file, indent=2, ensure_ascii=False)
json_file.close()
# categoryBig2categoryBigID
json_file = open(os.path.join(cf.JSON2, 'categoryBig2categoryBigID.json'), mode="w")
json.dump(categoryBig2categoryBigID, json_file, indent=2, ensure_ascii=False)
json_file.close()
# categoryMiddle2categoryMiddleID
json_file = open(os.path.join(cf.JSON2, 'categoryMiddle2categoryMiddleID.json'), mode="w")
json.dump(categoryMiddle2categoryMiddleID, json_file, indent=2, ensure_ascii=False)
json_file.close()

# rearrange id
def reSession(reviewerID):
    if reviewerID in reviewerID2sessionID:
        return reviewerID2sessionID[reviewerID]
    else:
        print('session is not recorded')
        return 'none'

def reItem(asin):
    if asin in asin2itemID:
        return asin2itemID[asin]
    else:
        print('item is not recorded')
        return 'none'

def reCateBig(category):
    if category in categoryBig2categoryBigID:
        return categoryBig2categoryBigID[category]
    else:
        print('categoryBig is not recorded')
        return 'none'

def reCateMiddle(category):
    if category in categoryMiddle2categoryMiddleID:
        return categoryMiddle2categoryMiddleID[category]
    else:
        print('categoryMiddle is not recorded')
        return 'none'

def priceInt(price):
    return int(price)
# rearrange id
data_all['sessionID'] = data_all.sessionID.map(reSession)
data_all['itemID'] = data_all.itemID.map(reItem)
data_all['priceLevel'] = data_all.priceLevel.map(priceInt)
data_all['categoryBig'] = data_all.categoryBig.map(reCateBig)
data_all['categoryMiddle'] = data_all.categoryMiddle.map(reCateMiddle)

data = data_all[['sessionID', 'time', 'period', 'super_sale', 'run_sale', 'gender', 'region', 'itemID', 'categoryMiddle', 'categoryBig', 'priceLevel']]
# filter item count (same as previous)
item_inter_num = pd.DataFrame(data.groupby(data['itemID']).count())
item_inter_num = item_inter_num.reset_index()[['sessionID', 'itemID']]
item_num = item_inter_num.rename(columns={'sessionID': 'item_num'})
data = pd.merge(data, item_num, how='left', on='itemID')
data = data[data['item_num'] > 9]
data = data[['sessionID', 'time', 'period', 'super_sale', 'run_sale', 'gender', 'region', 'categoryBig', 'categoryMiddle', 'itemID', 'priceLevel']]

# dict (sessionID:[itemID,itemID])
sess_all = {}
# dict (sessionID:[priceLevel, priceLevel])
price_all = {}
# dict (sessionID:[cateBig, cateBig])
cateBig_all = {}
# dict (sessionID:[cateMiddle, cateMiddle])
cateMiddle_all = {}
period_all = {}
super_sale_all = {}
run_sale_all = {}
gender_all = {}
region_all = {}
for _, row in data.iterrows():
    sess_id = row['sessionID']
    item_id = row['itemID']
    price = row['priceLevel']
    cateBig = row['categoryBig']
    cateMiddle = row['categoryMiddle']
    period = row['period']
    super_sale = row['super_sale']
    run_sale = row['run_sale']
    gender = row['gender']
    region = row['region']
    if sess_id in sess_all:
        sess_all[sess_id].append(item_id)
        price_all[sess_id].append(price)
        cateBig_all[sess_id].append(cateBig)
        cateMiddle_all[sess_id].append(cateMiddle)
    else:
        sess_all[sess_id] = []
        sess_all[sess_id].append(item_id)
        price_all[sess_id] = []
        price_all[sess_id].append(price)
        cateBig_all[sess_id] = []
        cateBig_all[sess_id].append(cateBig)
        cateMiddle_all[sess_id] = []
        cateMiddle_all[sess_id].append(cateMiddle)
        period_all[sess_id] = period
        super_sale_all[sess_id] = super_sale
        run_sale_all[sess_id] = run_sale
        gender_all[sess_id] = gender
        region_all[sess_id] = region

# sess_total = data['sessionID'].max()
sess_split = data[data['time'] == 102]['sessionID'].iloc[0]


# split_num = int(sess_total / 10 * 9) # train test split 9:1

tra_sess = dict()  # dict(session_id:[item_id,item_id,...])
tes_sess = dict()
tra_price = dict()  # dict(session_id:[price,price,...])
tes_price = dict()
tra_cateBig = dict()  # dict(session_id:[cateBig,cateBig,...])
tes_cateBig = dict()
tra_cateMiddle = dict()  # dict(session_id:[cateMiddle,cateMiddle,...])
tes_cateMiddle = dict()
tra_period = dict()
tes_period = dict()
tra_super_sale = dict()
tes_super_sale = dict()
tra_run_sale = dict()
tes_run_sale = dict()
tra_gender = dict()  # dict(session_id:[gender,gender,...])
tes_gender = dict()
tra_region = dict()  # dict(session_id:[region,region,...])
tes_region = dict()
for sess_temp in sess_all.keys():
    # 各セッションの item, price, category
    all_seqs = sess_all[sess_temp]
    all_price = price_all[sess_temp]
    all_cateBig = cateBig_all[sess_temp]
    all_cateMiddle = cateMiddle_all[sess_temp]
    period = period_all[sess_temp]
    super_sale = super_sale_all[sess_temp]
    run_sale = run_sale_all[sess_temp]
    gender = gender_all[sess_temp]
    region = region_all[sess_temp]
    # filter new session (less than 2)
    if len(all_seqs) < 2:
        continue
    # filter new session (more than 20)
    if len(all_seqs) > 20:
        all_seqs = all_seqs[:20]
        all_price = all_price[:20]
        all_cateBig = all_cateBig[:20]
        all_cateMiddle = all_cateMiddle[:20]
        period = period_all[sess_temp]
        super_sale = super_sale_all[sess_temp]
        run_sale = run_sale_all[sess_temp]
        gender = gender_all[sess_temp]
        region = region_all[sess_temp]
    # processed train, test
    if int(sess_temp) < sess_split:
        # train data
        tra_sess[sess_temp] = all_seqs
        tra_price[sess_temp] = all_price
        tra_cateBig[sess_temp] = all_cateBig
        tra_cateMiddle[sess_temp] = all_cateMiddle
        tra_period[sess_temp] = period
        tra_super_sale[sess_temp] = super_sale
        tra_run_sale[sess_temp] = run_sale
        tra_gender[sess_temp] = gender
        tra_region[sess_temp] = region
    else:
        # test data
        tes_sess[sess_temp] = all_seqs
        tes_price[sess_temp] = all_price
        tes_cateBig[sess_temp] = all_cateBig
        tes_cateMiddle[sess_temp] = all_cateMiddle
        tes_period[sess_temp] = period
        tes_super_sale[sess_temp] = super_sale
        tes_run_sale[sess_temp] = run_sale
        tes_gender[sess_temp] = gender
        tes_region[sess_temp] = region


item_dict = {}        # dict(old_itemID: new_itemID)
cateBig_dict = {}     # dict(old_cateBig: new_cateBig)
cateMiddle_dict = {}  # dict(old_cateMiddle: new_cateMiddle)
price_dict = {}       # dict(old_price: new_price)
super_sale_dict = {}
run_sale_dict = {}
gender_dict = {}
region_dict = {}

# tra_sess tra_price tra_cate
# Convert training sessions to sequences and renumber items to start from 1
# SR-GNN と同じ (各dict の value を list にまとめる)
def obtian_tra():
    train_seqs = []
    train_price = []
    train_cateBig = []
    train_cateMiddle = []
    train_period = []
    train_super_sale = []
    train_run_sale = []
    train_gender = []
    train_region = []
    item_ctr = 1
    price_ctr = 1
    cateBig_ctr = 1
    cateMiddle_ctr = 1
    super_sale_itr = 1
    run_sale_itr = 1
    gender_itr = 1
    region_itr = 1
    for s in tra_sess:
        seq = tra_sess[s]
        price_seq = tra_price[s]
        cateBig_seq = tra_cateBig[s]
        cateMiddle_seq = tra_cateMiddle[s]
        super_sale = tra_super_sale[s]
        run_sale = tra_run_sale[s]
        period = tra_period[s]
        gender = tra_gender[s]
        region = tra_region[s]
        outseq = []
        pri_outseq = []
        cateBig_outseq = []
        cateMiddle_outseq = []
        for i, p, cb, cm in zip(seq, price_seq, cateBig_seq, cateMiddle_seq):
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[int(i)] = item_ctr
                item_ctr += 1
            if p in price_dict:
                pri_outseq += [price_dict[p]]
            else:
                pri_outseq += [price_ctr]
                price_dict[int(p)] = price_ctr
                price_ctr += 1
            if cb in cateBig_dict:
                cateBig_outseq += [cateBig_dict[cb]]
            else:
                cateBig_outseq += [cateBig_ctr]
                cateBig_dict[int(cb)] = cateBig_ctr
                cateBig_ctr += 1
            if cm in cateMiddle_dict:
                cateMiddle_outseq += [cateMiddle_dict[cm]]
            else:
                cateMiddle_outseq += [cateMiddle_ctr]
                cateMiddle_dict[int(cm)] = cateMiddle_ctr
                cateMiddle_ctr += 1
        # super_sale, run_sale, gender, region はスカラー
        if super_sale in super_sale_dict:
            new_super_sale = super_sale_dict[super_sale]
        else:
            new_super_sale = super_sale_itr
            super_sale_dict[int(super_sale)] = super_sale_itr
            super_sale_itr += 1
        if run_sale in run_sale_dict:
            new_run_sale = run_sale_dict[run_sale]
        else:
            new_run_sale = run_sale_itr
            run_sale_dict[int(run_sale)] = run_sale_itr
            run_sale_itr += 1
        if gender in gender_dict:
            new_gender = gender_dict[gender]
        else:
            new_gender = gender_itr
            gender_dict[int(gender)] = gender_itr
            gender_itr += 1
        if region in region_dict:
            new_region = region_dict[region]
        else:
            new_region = region_itr
            region_dict[int(region)] = region_itr
            region_itr += 1
        if len(outseq) < 2:  # Doesn't occur
            print('session length is 1')
            continue
        train_seqs += [outseq]
        train_price += [pri_outseq]
        train_cateBig += [cateBig_outseq]
        train_cateMiddle += [cateMiddle_outseq]
        train_period.append(int(period))
        train_super_sale.append(new_super_sale)
        train_run_sale.append(new_run_sale)
        train_gender.append(new_gender)
        train_region.append(new_region)
    print("#train_session", len(train_seqs))
    print("#train_items", item_ctr - 1)
    print("#train_price", price_ctr - 1)
    print("#train_categoryBig", cateBig_ctr - 1)
    print("#train_categoryMiddle", cateMiddle_ctr - 1)
    print("#train_super_sale", super_sale_itr - 1)
    print("#train_run_sale", run_sale_itr - 1)
    print("#train_gender", gender_itr - 1)
    print("#train_region", region_itr - 1)
    # save dict
    # item_dict
    json_file = open(os.path.join(cf.JSON2, 'item_dict.json'), mode="w")
    json.dump(item_dict, json_file, indent=2, ensure_ascii=False)
    json_file.close()
    # price_dict
    json_file = open(os.path.join(cf.JSON2, 'price_dict.json'), mode="w")
    json.dump(price_dict, json_file, indent=2, ensure_ascii=False)
    json_file.close()
    # cateBig_dict
    json_file = open(os.path.join(cf.JSON2, 'cateBig_dict.json'), mode="w")
    json.dump(cateBig_dict, json_file, indent=2, ensure_ascii=False)
    json_file.close()
    # cateMiddle_dict
    json_file = open(os.path.join(cf.JSON2, 'cateMiddle_dict.json'), mode="w")
    json.dump(cateMiddle_dict, json_file, indent=2, ensure_ascii=False)
    json_file.close()
    # super_sale_dict
    json_file = open(os.path.join(cf.JSON2, 'super_sale_dict.json'), mode="w")
    json.dump(super_sale_dict, json_file, indent=2, ensure_ascii=False)
    json_file.close()
    # run_sale_dict
    json_file = open(os.path.join(cf.JSON2, 'run_sale_dict.json'), mode="w")
    json.dump(run_sale_dict, json_file, indent=2, ensure_ascii=False)
    json_file.close()
    # gender_dict
    json_file = open(os.path.join(cf.JSON2, 'gender_dict.json'), mode="w")
    json.dump(gender_dict, json_file, indent=2, ensure_ascii=False)
    json_file.close()
    # region_dict
    json_file = open(os.path.join(cf.JSON2, 'region_dict.json'), mode="w")
    json.dump(region_dict, json_file, indent=2, ensure_ascii=False)
    json_file.close()
    return train_seqs, train_price, train_cateBig, train_cateMiddle, train_period, train_super_sale, train_run_sale, train_gender, train_region


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_seqs = []
    test_price = []
    test_cateBig = []
    test_cateMiddle = []
    test_period = []
    test_super_sale = []
    test_run_sale = []
    test_gender = []
    test_region = []
    for s in tes_sess:
        outseq = []
        out_price = []
        out_cateBig = []
        out_cateMiddle = []
        p = tes_period[s]
        ss = tes_super_sale[s]
        rs = tes_run_sale[s]
        g = tes_gender[s]
        r = tes_region[s]
        for i, j, k, l in zip(tes_sess[s], tes_price[s], tes_cateBig[s], tes_cateMiddle[s]):
            # ignoring items that do not appear in training set
            if i in item_dict:
                outseq += [item_dict[i]]
                out_price += [price_dict[j]]
                out_cateBig += [cateBig_dict[k]]
                out_cateMiddle += [cateMiddle_dict[l]]
        if len(outseq) < 2:
            print('obtain test session length is 1')
            continue
        test_seqs += [outseq]
        test_price += [out_price]
        test_cateBig += [out_cateBig]
        test_cateMiddle += [out_cateMiddle]
        test_period.append(int(p))
        test_super_sale.append(super_sale_dict[ss])
        test_run_sale.append(run_sale_dict[rs])
        test_gender.append(gender_dict[g])
        test_region.append(region_dict[r])
    return test_seqs, test_price, test_cateBig, test_cateMiddle, test_period, test_super_sale, test_run_sale, test_gender, test_region

# split train seq and train target label
def process_seqs_no(iseqs, iprice, icateBig, icateMiddle):
    print("no data augment") # no data augmentation (like sr_gnn, gce_gnn)
    out_seqs = [] # train item seqs for each session
    out_price = [] # train price seqs
    out_cateBig = [] # train categoryBig seqs
    out_cateMiddle = [] # train categoryMiddle seqs
    labs = [] # target item
    for seq, pri, catb, catm in zip(iseqs, iprice, icateBig, icateMiddle):
        # target item (last item in a sequence)
        labs += [seq[-1]]
        out_seqs += [seq[:-1]]
        out_price += [pri[:-1]]
        out_cateBig += [catb[:-1]]
        out_cateMiddle += [catm[:-1]]
    return out_seqs, out_price, out_cateBig, out_cateMiddle, labs


tra_seqs, tra_pri, tra_catb, tra_catm, tra_pe, tra_ss, tra_rs, tra_ger, tra_reg = obtian_tra()
tes_seqs, tes_pri, tes_catb, tes_catm, tes_pe, tes_ss, tes_rs, tes_ger, tes_reg = obtian_tes()

tr_seqs, tr_pri, tr_catb, tr_catm, tr_labs = process_seqs_no(tra_seqs, tra_pri, tra_catb, tra_catm)
te_seqs, te_pri, te_catb, te_catm, te_labs = process_seqs_no(tes_seqs, tes_pri, tes_catb, tes_catm)

print('train sequence: ', tr_seqs[:5])
print('train price: ', tr_pri[:5]) # priceLevel
print('train categoryBig: ', tr_catb[:5])
print('train categoryMiddle: ', tr_catm[:5])
print('train lab: ', tr_labs[:5])

# Heterogeneous Hypergraph の接続行列作成の準備
def tomatrix(all_seqs, all_pri, all_cateBig, all_cateMiddle):
    price_item_dict = {} # dict(priceLevel:[item, item,...]) value: unique
    price_item = [] # dict values
    price_categoryBig_dict = {} # dict(priceLevel:[categoryBig, categoryBig,...])
    price_categoryBig = [] # dict values
    price_categoryMiddle_dict = {} # dict(priceLevel:[categoryMiddle, categoryMiddle,...])
    price_categoryMiddle = [] # dict values
    categoryBig_item_dict = {} # dict(categoryBig: [item, item,...])
    categoryMiddle_item_dict = {} # dict(categoryMiddle: [item, item,...])
    categoryBig_categoryMiddle_dict = {} # dict(categoryBig: [categoryMiddle, categoryMiddle,...])
    categoryBig_item = [] # dict values
    categoryMiddle_item = [] # dict values
    categoryBig_categoryMiddle = [] # dict values

    for s_seq, p_seq, cb_seq, cm_seq in zip(all_seqs, all_pri, all_cateBig, all_cateMiddle):
        # for each session (item, priceLevel, category)
        for i_temp, p_temp, cb_temp, cm_temp in zip(s_seq, p_seq, cb_seq, cm_seq):
            if p_temp not in price_item_dict:
                price_item_dict[p_temp] = []
            if p_temp not in price_categoryBig_dict:
                price_categoryBig_dict[p_temp] = []
            if p_temp not in price_categoryMiddle_dict:
                price_categoryMiddle_dict[p_temp] = []
            if cb_temp not in categoryBig_item_dict:
                categoryBig_item_dict[cb_temp] = []
            if cm_temp not in categoryMiddle_item_dict:
                categoryMiddle_item_dict[cm_temp] = []
            if cb_temp not in categoryBig_categoryMiddle_dict:
                categoryBig_categoryMiddle_dict[cb_temp] = []
            price_item_dict[p_temp].append(i_temp)
            price_categoryBig_dict[p_temp].append(cb_temp)
            categoryBig_item_dict[cb_temp].append(i_temp)
            price_categoryMiddle_dict[p_temp].append(cm_temp)
            categoryMiddle_item_dict[cm_temp].append(i_temp)
            categoryBig_categoryMiddle_dict[cb_temp].append(cm_temp)

    price_item_dict = dict(sorted(price_item_dict.items()))
    price_categoryBig_dict = dict(sorted(price_categoryBig_dict.items()))
    price_categoryMiddle_dict = dict(sorted(price_categoryMiddle_dict.items()))
    categoryBig_item_dict = dict(sorted(categoryBig_item_dict.items()))
    categoryMiddle_item_dict = dict(sorted(categoryMiddle_item_dict.items()))
    categoryBig_categoryMiddle_dict = dict(sorted(categoryBig_categoryMiddle_dict.items()))
    
    print("#price", len(price_item_dict))
    print("#categoryBig", len(categoryBig_item_dict))
    print("#categoryMiddle", len(categoryMiddle_item_dict))
    price_item = list(price_item_dict.values())
    price_categoryBig = list(price_categoryBig_dict.values())
    categoryBig_item = list(categoryBig_item_dict.values())
    price_categoryMiddle = list(price_categoryMiddle_dict.values())
    categoryMiddle_item = list(categoryMiddle_item_dict.values())
    categoryBig_categoryMiddle = list(categoryBig_categoryMiddle_dict.values())
    
    return price_item, price_categoryBig, categoryBig_item, price_categoryMiddle, categoryMiddle_item, categoryBig_categoryMiddle


def data_masks(all_sessions):
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
        results = (data, indices, indptr)
    return results

# Heterogeneous Hypergraph の接続行列は train, test 全てのデータから作成する
# tra_pi, tra_pcb, tra_cbi, tra_pcm, tra_cmi, tra_cbcm = tomatrix(tra_seqs + tes_seqs, tra_pri + tes_pri, tra_catb + tes_catb, tra_catm + tes_catm)
tra_pi, tra_pcb, tra_cbi, tra_pcm, tra_cmi, tra_cbcm = tomatrix(tra_seqs, tra_pri, tra_catb, tra_catm)
# concat
tra = (
tr_seqs, tr_pri, data_masks(tr_seqs), data_masks(tr_pri), data_masks(tra_pi), data_masks(tra_pcb), data_masks(tra_cbi), data_masks(tra_pcm), data_masks(tra_cmi), data_masks(tra_cbcm), tra_pe, tra_ss, tra_rs, tra_ger, tra_reg, 
tr_labs)
tes = (
te_seqs, te_pri, data_masks(te_seqs), data_masks(te_pri), data_masks(tra_pi), data_masks(tra_pcb), data_masks(tra_cbi), data_masks(tra_pcm), data_masks(tra_cmi), data_masks(tra_cbcm), tes_pe, tes_ss, tes_rs, tes_ger, tes_reg,
te_labs)

all = 0
for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('#interactions: ', all) # num of all items
print('#train_session: ', (len(tra_seqs)))
print('#test_session: ', (len(tes_seqs)))
print('#all_session: ', (len(tra_seqs) + len(tes_seqs))) # num of sessions
print('sequence average length: ', all / (len(tra_seqs) + len(tes_seqs) * 1.0)) # avg session length

pickle.dump(tra, open(cf.train_path, 'wb'))
pickle.dump(tes, open(cf.test_path, 'wb'))
pickle.dump(tra_seqs, open(cf.all_train_seq_path, 'wb')) # global graph の構築で必要
pickle.dump(tes_seqs, open(cf.all_test_seq_path, 'wb'))
pickle.dump(tra_pri, open(cf.all_train_price_seq_path, 'wb')) # global graph の構築で必要
pickle.dump(tes_pri, open(cf.all_test_price_seq_path, 'wb'))
print("dataset: ", cf.age_range)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("done")