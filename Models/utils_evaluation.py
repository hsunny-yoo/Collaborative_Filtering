import numpy as np
from sklearn.metrics import mean_squared_error
import math


def hr(reco, gt, k):
    """
    :param reco: (2d list) item ids each users (sorted descending)
    :param gt: (2d list) item id each users (one item)
                         ex: [[1], [3], [], [1], ...]
    :return: Hit Ratio @ k
    """
    hit_ratio = 0
    cnt = 0

    for i in range(len(gt)):
        if not len(gt[i]) >0:
            continue

        if gt[i][0] in reco[i][:k]:
            hit_ratio += 1
        cnt += 1

    hit_ratio /= cnt
    return hit_ratio


def mse(pred, gt):
    """
    :param pred: (1d list) predicted values for testset
    :param gt: (1d list) ground truth values for testset
    :return: MSE
    """
    return mean_squared_error(pred, gt)


def mre(pred, true):
    mre = 0
    cnt = 0
    for i in range(true.shape[0]):
        if true[i].shape[0] == len(set(true[i])):
            continue
        argsort_pred = np.argsort(pred[i]).tolist()
        argsort_true = np.argsort(true[i]).tolist()

        if argsort_pred != argsort_true:
            mre += 1
        cnt += 1

    mre /= cnt
    return mre


def map(reco, gt, k):
    """
    :param reco: (2d list) item ids each users (sorted descending)
    :param gt: (2d list) item ids each users
                         ex: [[1], [3,5], [], [1,3,5], ...]
    :param k:
    :return: Average of MAP@K for all users
    """
    map = 0
    cnt = 0
    for i in range(len(gt)):
        if len(gt[i]) == 0:
            continue

        top_k_items = reco[i][:k]

        ap = 0
        for j in range(1, k + 1):
            numer = len(set(top_k_items[:j]).intersection(set(gt[i])))
            denom = j
            ap += numer / denom
        ap /= min(k, len(gt[i]))

        map += ap
        cnt += 1

    map /= cnt
    return map


def auc(reco, gt):
    """
    :param reco: (2d list) item ids each users (sorted descending)
    :param gt: (2d list) item id each users (one item)
                         ex: [[1], [3], [], [1], ...]
    :return:
    """
    auc = 0
    cnt = 0
    for i in range(len(gt)):
        if not len(gt[i]) > 0:
            continue

        if gt[i][0] not in reco[i]:
            cnt += 1
            continue

        gt_idx = reco[i].index(gt[i][0])
        auc_ = (len(reco[i]) - gt_idx)/len(reco[i])

        auc += auc_
        cnt += 1

    auc /= cnt
    return auc


def ndcg(reco, gt, k):
    """
    :param reco: (2d list) item ids each users (sorted descending)
    :param gt: (2d list) item ids each users (sorted descending if explicit feedback)
                         ex: [[1], [3,5], [], [1,3,5], ...]
    :param k:
    :return: Average of NDCG@k for all users
    """
    count = 0
    ndcg = 0
    for i in range(len(gt)):
        if not len(gt[i]) > 0:
            continue

        idcg = sum([1.0 / math.log(i + 2, 2) for i in range(len(gt[i]))])
        dcg = 0.0
        for j, r in enumerate(reco[i][:k]):
            if r not in gt[i]:
                continue

            gt_index = gt[i].index(r)
            if j != gt_index:
                rel = 0.7
            else:
                rel = 1.0
            dcg += rel / math.log(i + 2, 2)

        ndcg_ = dcg / idcg

        ndcg += ndcg_
        count += 1

    ndcg /= count

    return ndcg

def acc():
    pass