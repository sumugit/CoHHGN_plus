import pickle
import json
adj = pickle.load(open('/workspace/datasets/object/20_35/adj_12.pkl', 'rb'))
weight = pickle.load(open('/workspace/datasets/object/20_35/num_12.pkl', 'rb'))
with open('/workspace/datasets/json/item_id_to_node_id.json') as f:
    id_to_id = json.load(f)
with open('/workspace/datasets/json/category_to_id_20_35.json') as f:
    category_to_id = json.load(f)
node_id = int(input())
# print input item
for k, v in id_to_id.items():
    if v == node_id:
        item_id = int(k)
        break
for k, v in category_to_id.items():
    if v == item_id:
        print(k)
        break
print('########################################')
item_ids = []
for id in adj[node_id]:
    for k, v in id_to_id.items():
        if v == id:
            item_ids.append(int(k))
            break
i = 0
for id in item_ids:
    for k, v in category_to_id.items():
        if v == id:
            print(k, weight[node_id][i])
            i += 1
            break