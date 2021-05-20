#!/user/bin/python
import random
import collections
import numpy as np

np.random.seed(0)
random.seed(0)


# def sample_neg_items_for_u_test(user_dict, test_dict, user_id, n_sample_neg_items):
#     pos_items = user_dict[user_id]
#     pos_items_2 = test_dict[user_id]
#
#     sample_neg_items = []
#     while True:
#         if len(sample_neg_items) == n_sample_neg_items:
#             break
#         neg_item_id = np.random.randint(low=0, high=14284, size=1)[0]
#         if neg_item_id not in pos_items and neg_item_id not in pos_items_2 and neg_item_id not in sample_neg_items:
#             sample_neg_items.append(neg_item_id)
#     return sample_neg_items


# train_rate = 0.8

# train = []

# dict = collections.defaultdict(list)
# train_dict = collections.defaultdict(list)
# test_dict = collections.defaultdict(list)

#
# data = []
# with open('data/yelp/yelp.train.rating', 'r') as infile:
#     for line in infile.readlines():
#         inter = [int(i) for i in line.strip().split('\t')]
#         data.append([inter[0], inter[1], 5])
#         dict[inter[0]].append(inter[1])
#         # train_dict[inter[0]].append(inter[1])
#
# with open('data/yelp/yelp.test.rating', 'r') as infile:
#     for line in infile.readlines():
#         inter = [int(i) for i in line.strip().split()]
#         user, item_ids = inter[0], inter[1:]
#         item_ids = list(set(item_ids))
#         for item in item_ids:
#             data.append([user, item, 5])
#             dict[user].append(item)
#             test_dict[inter[0]].append(str(item))

# user_ids_batch = list(dict.keys())
# neg_dict = collections.defaultdict(list)

# for u in user_ids_batch:
#     for _ in dict[u]:
#         nl = sample_neg_items_for_u_test(train_dict, test_dict, u, 100)
#         for l in nl:
#             train.append([str(u), str(l), '0'])

# random.shuffle(data)
# random.shuffle(test)


# fw1 = open('data/ml-100k/yelp.train.rating', 'w')
# fw2 = open('data/ml-100k/ml-100k.test.rating', 'w')
# fw3 = open('data/yelp/yelp.test.negative', 'w')
#
#
# for user in test_dict:
#     fw3.write('(' + str(user) + ',' + ','.join(test_dict[user]) + ')')
#     for i in range(1, 14285):
#         if i not in dict[user]:
#             fw3.write(' ' + str(i))
#     fw3.write('\n')



# with open('../data/ub_' + str(train_rate) + '.train2', 'w') as trainfile, \
#         open('../data/ub_' + str(train_rate) + '.test2', 'w') as testfile:
#     for r in train:
#         trainfile.write('\t'.join(r) + '\n')
#     for r in test:
#         testfile.write('\t'.join(r) + '\n')


data = []
# with open('data/yelp/yelp.bpr.user_embedding', 'r') as infile:
#     for line in infile.readlines():
#         inter = [float(i) for i in line.strip().split('\t')]
#         data.append([inter[0], inter[1], 5])
#         dict[inter[0]].append(inter[1])
#         # train_dict[inter[0]].append(inter[1])


fw1 = open('data/yelp/yelp.bpr.item_embedding1', 'w')
fw2 = open('data/yelp/yelp.bpr.user_embedding1', 'w')

with open('data/yelp/yelp.bpr.item_embedding', 'r') as infile:
    for line in infile.readlines():
        inter = [float(i) for i in line.strip().split()]
        id, embedding = int(inter[0]), inter[1:]
        fw1.write(str(id + 1) + ' ')
        for embed in embedding:
            fw1.write(str(embed) + " ")
        fw1.write('\n')


with open('data/yelp/yelp.bpr.user_embedding', 'r') as infile:
    for line in infile.readlines():
        inter = [float(i) for i in line.strip().split()]
        id, embedding = int(inter[0]), inter[1:]
        fw2.write(str(id + 1) + ' ')
        for embed in embedding:
            fw2.write(str(embed) + " ")
        fw2.write('\n')
