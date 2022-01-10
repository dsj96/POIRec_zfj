import numpy as np

import tqdm


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users



def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def accuracy_at_k(actual, predicted, topk):
    test_num = 0.0
    hit_num = 0.0
    num_users = len(predicted)

    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])

        if len(act_set) != 0:
            if len(act_set & pred_set) != 0:
                hit_num = hit_num + 1
            test_num += 1
    return hit_num / test_num

def ndcg_at_k(actual, predicted, topk):
    sum_ndcg_score = 0.0
    num_users = len(predicted)
    true_users = 0
    # idcg score
    idcg_score = 0.0
    for index in range(topk):
        idcg_score += np.reciprocal(np.log2(index+2))

    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            true_users += 1
            # dcg score
            dcg_score = 0.0
            for (index,cand_poi) in enumerate(predicted[i][:topk]):
                if cand_poi in act_set:
                    dcg_score += np.reciprocal(np.log2(index+2))
            ndcg_score = dcg_score/idcg_score
            sum_ndcg_score += ndcg_score

    return sum_ndcg_score/true_users

def hit_ratio_at_k(actual, predicted, topk):
    num_users = len(predicted)
    true_users = 0
    sum_num_of_hit = 0
    sum_GT = 0

    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            true_users += 1
            sum_num_of_hit += len(act_set & pred_set)
            sum_GT += len(act_set)

    return sum_num_of_hit/sum_GT




# def evaleate(candidate_pois_array, LSTM_test_target_poi_list, node_type, final_model2):
def evaleate(score_matrix_index, train_user_dict, test_user_dict):
    test_users = list(test_user_dict.keys())
    num_users, num_pois = score_matrix_index.shape
    candidate_poi_dict = dict()
    accuracy = []
    precision = []
    recall = []
    ndcg = []
    hit_ratio = []
    MAP = []
    for user in test_user_dict.keys():
        for i in range(num_pois):
            # 用户在friendship.txt中，没有在train test.txt中
            try:
                index = np.argwhere(score_matrix_index[user-1,:] == i) # TODO 降序index=0 socre 最大，user的0，1开始注意
                if (index+num_users+1).tolist()[0][0] not in train_user_dict[user]:
                    if user not in candidate_poi_dict.keys():
                        candidate_poi_dict[user] = [(index+num_users+1).tolist()[0][0]]
                    else:
                        candidate_poi_dict[user].append((index+num_users+1).tolist()[0][0])
                if len(candidate_poi_dict[user])==20:
                    break
            except:
                pass

    pre_list, target_list = [], []
    for user, trg in candidate_poi_dict.items():
        pre_list.append(candidate_poi_dict[user])
        target_list.append(test_user_dict[user])

    with open('train_log.txt', 'w', encoding='utf-8') as f_log:

        for k in tqdm.tqdm([1, 5, 10, 15, 20]):
            accuracy.append(accuracy_at_k(target_list, pre_list, k))
            precision.append(precision_at_k(target_list, pre_list, k))
            recall.append(recall_at_k(target_list, pre_list, k))
            ndcg.append(ndcg_at_k(target_list, pre_list, k))
            hit_ratio.append(hit_ratio_at_k(target_list, pre_list, k))
            MAP.append(mapk(target_list, pre_list, k))
        print('accuracy: ', accuracy)
        print('precision: ', precision)
        print('recall: ', recall)
        print('ndcg: ', ndcg)
        print('hit_ratio: ', hit_ratio)
        print('MAP: ', MAP)
        f_log.write('accuracy: '+ str(accuracy)+'\n')
        f_log.write('precision: '+ str(precision)+'\n')
        f_log.write('recall: '+  str(recall)+ '\n')
        f_log.write('ndcg: '+ str(ndcg)+'\n')
        f_log.write('hit_ratio: '+ str(hit_ratio)+ '\n')
        f_log.write('MAP: '+ str(MAP)+'\n')

    return accuracy, precision, recall, ndcg, hit_ratio, MAP


