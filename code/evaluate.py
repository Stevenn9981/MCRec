'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np
import random
from time import time
import scipy.sparse as sp
import gc

# from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None
_num_users = None
_num_items = None
_path_umtm = None
_path_umum = None
_path_umtmum = None
_path_uuum = None
_path_nums = None
_timestamps = None
_length = None

_user_feature = None
_item_feature = None
_type_feature = None
_features = None


def evaluate_model(model, user_feature, item_feature, type_feature, num_users, num_items,
                   path_umtm,
                   path_umum,
                   path_umtmum,
                   path_uuum,
                   path_nums, timestamps, length, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _num_users
    global _num_items
    global _path_umtm
    global _path_umum
    global _path_umtmum
    global _path_uuum
    global _path_nums
    global _timestamps
    global _length

    global _user_feature
    global _item_feature
    global _features

    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    _num_users = num_users
    _num_items = num_items
    _path_umtm = path_umtm
    _path_umum = path_umum
    _path_umtmum = path_umtmum
    _path_uuum = path_uuum
    _path_nums = path_nums
    _timestamps = timestamps
    _length = length

    _user_feature = user_feature
    _item_feature = item_feature
    _features = [user_feature, item_feature, type_feature]
    ps, rs, ndcgs = [], [], []
    if (num_thread > 1):  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    # print("len test: ", len(_testRatings))

    # for idx in range(len(_testRatings)):
    for idx in random.sample(range(len(_testRatings)), 50):
        (p, r, ndcg) = eval_one_rating(idx)
        ps.extend(p)
        rs.extend(r)
        ndcgs.extend(ndcg)

    # ps, rs, ndcgs = eval()
    return (ps, rs, ndcgs)


# def eval():
#     items = []
#     user_input = []
#     item_input = []
#     map_item_score = {}
#     users = []
#     for idx, rating in enumerate(_testRatings):
#         items += _testNegatives[idx]
#         u = rating[0]
#         gtItems = rating[1:]
#         # pItems += gtItems
#         # items.append(gtItem)
#         items += gtItems
#         users += [u] * (len(_testNegatives[idx]) + len(gtItems))
#         user_input += [u] * 101 * len(gtItems)
#         for pItem in gtItems:
#             item_input += [pItem]
#             nItems = random.sample(_testNegatives[idx], 100)
#             item_input += nItems
#
#     umtm_input = np.zeros((len(items), _path_nums[0], _timestamps[0], _length))
#     umum_input = np.zeros((len(items), _path_nums[1], _timestamps[1], _length))
#     umtmum_input = np.zeros((len(items), _path_nums[2], _timestamps[2], _length))
#     uuum_input = np.zeros((len(items), _path_nums[3], _timestamps[3], _length))
#
#     # for idx, rating in enumerate(_testRatings):
#     #     # Get prediction scores
#     #     u = rating[0]
#     #     items = _testNegatives[idx]
#     #     gtItems = rating[1:]
#     #     items += gtItems
#
#     time1 = time()
#     print('Timing start!')
#
#     k = 0
#     for index, i in enumerate(items):
#
#         # user_input.append(u)
#         u = users[index]
#         # item_input.append(i)
#
#         if (u, i) in _path_umtm:
#             for p_i in range(len(_path_umtm[(u, i)])):
#                 for p_j in range(len(_path_umtm[(u, i)][p_i])):
#                     type_id = _path_umtm[(u, i)][p_i][p_j][0]
#                     index = _path_umtm[(u, i)][p_i][p_j][1]
#                     if type_id == 1:
#                         umtm_input[k][p_i][p_j] = _user_feature[index]
#                     elif type_id == 2:
#                         umtm_input[k][p_i][p_j] = _item_feature[index]
#
#         if (u, i) in _path_umum:
#             for p_i in range(len(_path_umum[(u, i)])):
#                 for p_j in range(len(_path_umum[(u, i)][p_i])):
#                     type_id = _path_umum[(u, i)][p_i][p_j][0]
#                     index = _path_umum[(u, i)][p_i][p_j][1]
#                     if type_id == 1:
#                         umum_input[k][p_i][p_j] = _user_feature[index]
#                     elif type_id == 2:
#                         umum_input[k][p_i][p_j] = _item_feature[index]
#
#         if (u, i) in _path_umtmum:
#             for p_i in range(len(_path_umtmum[(u, i)])):
#                 for p_j in range(len(_path_umtmum[(u, i)][p_i])):
#                     type_id = _path_umtmum[(u, i)][p_i][p_j][0]
#                     index = _path_umtmum[(u, i)][p_i][p_j][1]
#                     if type_id == 1:
#                         umtmum_input[k][p_i][p_j] = _user_feature[index]
#                     elif type_id == 2:
#                         umtmum_input[k][p_i][p_j] = _item_feature[index]
#         if (u, i) in _path_uuum:
#             for p_i in range(len(_path_uuum[(u, i)])):
#                 for p_j in range(len(_path_uuum[(u, i)][p_i])):
#                     type_id = _path_uuum[(u, i)][p_i][p_j][0]
#                     index = _path_uuum[(u, i)][p_i][p_j][1]
#                     if type_id == 1:
#                         uuum_input[k][p_i][p_j] = _user_feature[index]
#                     elif type_id == 2:
#                         uuum_input[k][p_i][p_j] = _item_feature[index]
#         k += 1
#
#     print('path_input process time: ', time() - time1)
#     print(len(item_input))
#     print(len(user_input))
#     # print umtm_input.shape
#     predictions = _model.predict(
#         [np.array(user_input), np.array(item_input), umtm_input, umum_input, umtmum_input, uuum_input],
#         batch_size=256, verbose=0)
#     # print atten.shape
#
#     print('Prediction time: ', time() - time1)
#
#     hs = []
#     rs = []
#     ns = []
#     for i in range(len(item_input)):
#         user = user_input[i]
#         item = item_input[i]
#         map_item_score[(user, item)] = predictions[i]
#     # items.pop()
#     # Evaluate top rank list
#
#     for i in range(0, len(item_input), 101):
#         user = user_input[i]
#         pItem = item_input[i]
#         preItems = item_input[i:i + 101]
#         item_score = dict()
#         for Item in preItems:
#             item_score[Item] = map_item_score[(user, Item)]
#         ranklist = heapq.nlargest(_K, item_score, key=item_score.get)
#
#         pItem = preItems[50]
#         print(pItem, ranklist)
#         p = getHitRatio(ranklist, [pItem])
#         r = getR(ranklist, [pItem])
#         ndcg = getNDCG(ranklist, [pItem])
#         hs.append(p)
#         rs.append(r)
#         ns.append(ndcg)
#
#     print('Sorting time: ', time() - time1)
#     print(hs)
#     return (hs, rs, ns)


def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItems = rating[1:]
    # items.append(gtItem)
    items += gtItems
    # Get prediction scores
    map_item_score = {}
    user_input = []
    item_input = []
    umtm_input = np.zeros((len(items), _path_nums[0], _timestamps[0], _length))
    umum_input = np.zeros((len(items), _path_nums[1], _timestamps[1], _length))
    umtmum_input = np.zeros((len(items), _path_nums[2], _timestamps[2], _length))
    uuum_input = np.zeros((len(items), _path_nums[3], _timestamps[3], _length))

    time1 = time()
    # print('Timing start!')

    k = 0
    for i in items:

        user_input.append(u)
        item_input.append(i)

        if (u, i) in _path_umtm:
            for p_i in range(len(_path_umtm[(u, i)])):
                for p_j in range(len(_path_umtm[(u, i)][p_i])):
                    type_id = _path_umtm[(u, i)][p_i][p_j][0]
                    index = _path_umtm[(u, i)][p_i][p_j][1]
                    if type_id == 1:
                        umtm_input[k][p_i][p_j] = _user_feature[index]
                    elif type_id == 2:
                        umtm_input[k][p_i][p_j] = _item_feature[index]

        if (u, i) in _path_umum:
            for p_i in range(len(_path_umum[(u, i)])):
                for p_j in range(len(_path_umum[(u, i)][p_i])):
                    type_id = _path_umum[(u, i)][p_i][p_j][0]
                    index = _path_umum[(u, i)][p_i][p_j][1]
                    if type_id == 1:
                        umum_input[k][p_i][p_j] = _user_feature[index]
                    elif type_id == 2:
                        umum_input[k][p_i][p_j] = _item_feature[index]

        if (u, i) in _path_umtmum:
            for p_i in range(len(_path_umtmum[(u, i)])):
                for p_j in range(len(_path_umtmum[(u, i)][p_i])):
                    type_id = _path_umtmum[(u, i)][p_i][p_j][0]
                    index = _path_umtmum[(u, i)][p_i][p_j][1]
                    if type_id == 1:
                        umtmum_input[k][p_i][p_j] = _user_feature[index]
                    elif type_id == 2:
                        umtmum_input[k][p_i][p_j] = _item_feature[index]
        if (u, i) in _path_uuum:
            for p_i in range(len(_path_uuum[(u, i)])):
                for p_j in range(len(_path_uuum[(u, i)][p_i])):
                    type_id = _path_uuum[(u, i)][p_i][p_j][0]
                    index = _path_uuum[(u, i)][p_i][p_j][1]
                    if type_id == 1:
                        uuum_input[k][p_i][p_j] = _user_feature[index]
                    elif type_id == 2:
                        uuum_input[k][p_i][p_j] = _item_feature[index]
        k += 1

    # print('path_input process time: ', time() - time1)
    # print(len(item_input))
    # print umtm_input.shape
    predictions = _model.predict(
        [np.array(user_input), np.array(item_input), umtm_input, umum_input, umtmum_input, uuum_input],
        batch_size=256, verbose=0)
    # print atten.shape

    # print('Prediction time: ', time() - time1)

    hs = []
    rs = []
    ns = []
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    # items.pop()
    # Evaluate top rank list
    for pItem in gtItems:
        nItems = random.sample(_testNegatives[idx], 100)
        item_score = dict()
        for nItem in nItems:
            item_score[nItem] = map_item_score[nItem]
        item_score[pItem] = map_item_score[pItem]
        ranklist = heapq.nlargest(_K, item_score, key=item_score.get)

        p = getHitRatio(ranklist, [pItem])
        r = getR(ranklist, [pItem])
        ndcg = getNDCG(ranklist, [pItem])
        hs.append(p)
        rs.append(r)
        ns.append(ndcg)

    # print('Sorting time: ', time() - time1)
    return (hs, rs, ns)


def getP(ranklist, gtItems):
    p = 0
    for item in ranklist:
        if item in gtItems:
            p += 1
    return p * 1.0 / len(ranklist)


def getR(ranklist, gtItems):
    r = 0
    for item in ranklist:
        if item in gtItems:
            r += 1
    return r * 1.0 / len(gtItems)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item in gtItem:
            return 1
    return 0


def getDCG(ranklist, gtItems):
    dcg = 0.0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item in gtItems:
            dcg += 1.0 / math.log(i + 2)
    return dcg


def getIDCG(ranklist, gtItems):
    idcg = 0.0
    i = 0
    for item in ranklist:
        if item in gtItems:
            idcg += 1.0 / math.log(i + 2)
            i += 1
    return idcg


def getNDCG(ranklist, gtItems):
    dcg = getDCG(ranklist, gtItems)
    idcg = getIDCG(ranklist, gtItems)
    if idcg == 0:
        return 0
    return dcg / idcg

# def getNDCG(ranklist, gtItems):
#    for i in range(len(ranklist)):
#        item = ranklist[i]
#        if item == gtItems:
#            return math.log(2) / math.log(i+2)
#    return 0
