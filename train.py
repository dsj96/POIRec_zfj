import argparse
from utils import *
from LINE_Model import MultiViewModel
from tqdm import trange
import torch
import torch.optim as optim
import sys
from sklearn import  cluster
import pickle
import metrics
# 关闭主机和设备间的异步执行，获取更多的输出信息
# https://blog.csdn.net/az9996/article/details/109245721?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164154606016780255242085%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=164154606016780255242085&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-109245721.first_rank_v2_pc_rank_v29&utm_term=ytorch_1556684464974%2Fwork%2Faten%2Fsrc%2FTHC%2FTHCTensorIndex.cu%3A362%3A+block%3A+%5B36%2C0%2C0%5D+%3C+srcSelectDimSize%60+failed.&spm=1018.2226.3001.4187
import os
from metrics import evaleate

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 1.设置模型参数
# 2.读图，存点和边并做归一化3.计算点和边的alias table
# 4.Line模型实现
# 5.模型按边训练以及负采样
# 6.结果展示和可视化
if __name__ == "__main__":
    # 1. # 设置模型参数；读图，存点和边并做归一化
    # 1）设置模型参数设置模型超参数，如1st order，2nd order，负样本数量（K），embedding维度，batch、epoch、learning rate等
    # 2）输入输出
    # 输入文件./data/weighted.karate.edgelist
    # 输出文件./model.pt
    parser = argparse.ArgumentParser()
    # 输入文件
    # parser.add_argument("-g", "--graph_path", type=str, default='./data/data02/weighted.karate.edgelist')
    parser.add_argument("-g", "--graph_path", type=str, default='./data/data04/')
    # 模型信息输出文件
    parser.add_argument("-save", "--save_path", type=str, default='./saved_model/model.pt')
    # 模型损失函数值输出文件
    parser.add_argument("-lossdata", "--lossdata_path", type=str, default='./saved_model/loss.pkl')

    # Hyperparams.超参数
    # 论文中的1st order，2nd order
    parser.add_argument("-order", "--order", type=int, default=2)
    # 负样本个数
    parser.add_argument("-neg", "--negsamplesize", type=int, default=5)
    # embedding维度
    parser.add_argument("-dim", "--dimension", type=int, default=128)
    # batchsize大小
    parser.add_argument("-batchsize", "--batchsize", type=int, default=10000)
    parser.add_argument("-epochs", "--epochs", type=int, default=2)
    # 学习率
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.025)  # As starting value in paper
    # 负采样指数值
    parser.add_argument("-negpow", "--negativepower", type=float, default=0.75)

    parser.add_argument("-seed", '--seed', type=int, default=0, help='Random seed.')

    args = parser.parse_args()

    # 2.读图，存点和边并做归一化
    # 1）读图自己实现的makeDist函数，在utils.py中
    # Create dict of distribution when opening file
    # 读图，函数在utils.py中
    edgedistdict_uu, nodedistdict_uu, weights_uu, nodedegrees_uu, maxindex_uu = makeDist(args.graph_path + 'G_uu.txt', args.negativepower)
    edgedistdict_up, nodedistdict_up, weights_up, nodedegrees_up, maxindex_up = makeDist(args.graph_path + 'G_up.txt', args.negativepower)
    edgedistdict_pp, nodedistdict_pp, weights_pp, nodedegrees_pp, maxindex_pp = makeDist(args.graph_path + 'G_pp.txt', args.negativepower)

    maxindex_uu = 3554
    # 3. 计算点和边的alias table
    # 构建alias table，达到O(1)的采样效率
    edgesaliassampler_uu = VoseAlias(edgedistdict_uu)
    nodesaliassampler_uu = VoseAlias(nodedistdict_uu)

    edgesaliassampler_up = VoseAlias(edgedistdict_up)
    nodesaliassampler_up = VoseAlias(nodedistdict_up)

    edgesaliassampler_pp = VoseAlias(edgedistdict_pp)
    nodesaliassampler_pp = VoseAlias(nodedistdict_pp)

    # LINE模型实现
    # 每次训练batch size大小的边数量
    batchrange_uu = int(len(edgedistdict_uu) / args.batchsize)
    batchrange_up = int(len(edgedistdict_up) / args.batchsize)
    batchrange_pp = int(len(edgedistdict_pp) / args.batchsize)
    batchrange  = max(batchrange_uu, batchrange_up, batchrange_pp)

    print('maxindex_uu = ', maxindex_uu)
    print('maxindex_up = ', maxindex_up)
    print('maxindex_pp = ', maxindex_pp)

    set_seed(args.seed)
    train_users_history = read_history(args.graph_path + 'train_checkin_file.txt', '\t')
    model = MultiViewModel(maxindex_uu+1, maxindex_up+1, maxindex_pp+1, embed_dim=args.dimension, order=args.order)

    # SGD优化，nesterov是对momentum的改进，是在momentum向量的终端再更新梯度。
    opt = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    # 选用gpu或cpu训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    lossdata = {"it": [], "loss": []}
    it = 0

    # print("\nTraining on {}...\n".format(device))
    # 共训练epoch次数
    for epoch in range(args.epochs):
        print("Epoch {}".format(epoch))
        # 每次训练组数：batchsize
        for b in trange(batchrange):
            # edgesaliassampler是实现alias building的VoseAlias类，这里采样出batchsize条边
            samplededges_uu = edgesaliassampler_uu.sample_n(args.batchsize)
            samplededges_up = edgesaliassampler_up.sample_n(args.batchsize)
            samplededges_pp = edgesaliassampler_pp.sample_n(args.batchsize)
            # samplededges_uu = edgesaliassampler_uu.sample_n(len(edgedistdict_uu))
            # samplededges_up = edgesaliassampler_up.sample_n(len(edgedistdict_up))
            # samplededges_pp = edgesaliassampler_pp.sample_n(len(edgedistdict_pp))

            # 存makeData是utils.py中的函数，为每条边采样出K条负样本边存每一条格式是（node i，node j，negative nodes...）
            batch_uu = list(makeData(samplededges_uu, args.negsamplesize, weights_uu, nodedegrees_uu, nodesaliassampler_uu))
            batch_up = list(makeData(samplededges_up, args.negsamplesize, weights_up, nodedegrees_up, nodesaliassampler_up))
            batch_pp = list(makeData(samplededges_pp, args.negsamplesize, weights_pp, nodedegrees_pp, nodesaliassampler_pp))

            # 转换成tensor格式
            batch_uu = torch.LongTensor(batch_uu)
            batch_up = torch.LongTensor(batch_up)
            batch_pp = torch.LongTensor(batch_pp)
            # 把一个batch的数据打印出来是这样：
            # tensor([[3, 8 14, 14, 24, 2, 32],
            #         [25, 32, 14, 9, 4, 24, 23],
            #         [1, 14, 32, 1, 25, 27, 16],
            #         [26, 32, 30, 4, 14, 7, 4],
            #         [25, 32, 25, 14, 20, 14, 27]])

            # 取第0列就是起始点
            v_i_uu = batch_uu[:, 0]
            v_i_up = batch_up[:, 0]
            v_i_pp = batch_pp[:, 0]
            # 取第1列就是终点
            v_j_uu = batch_uu[:, 1]
            v_j_up = batch_up[:, 1]
            v_j_pp = batch_pp[:, 1]
            # 取后面5列就是负样本
            negsamples_uu = batch_uu[:, 2:]
            negsamples_up = batch_up[:, 2:]
            negsamples_pp = batch_pp[:, 2:]
            # 在做BP之前将gradients置因为是梯度累加的
            model.zero_grad()

            # 模型实现部分
            loss, fused_emb = model(v_i_uu, v_j_uu, negsamples_uu, v_i_up, v_j_up, negsamples_up, v_i_pp, v_j_pp, negsamples_pp, train_users_history, device)
            # 计算梯度
            loss.backward()
            # 根据梯度值更新参数值
            opt.step()

            lossdata["loss"].append(loss.item())
            lossdata["it"].append(it)
            it += 1

    print("\nDone training, saving model to {}".format(args.save_path))
    torch.save(model, "{}".format(args.save_path))

    # calculate the simi score
    # score_matrix = torch.mm(fused_emb, fused_emb.t())[1:maxindex_uu+1, maxindex_uu+1:]               # [user_num*poi_num] 10*24
    # TODO:  正序反序 有差别
    score_matrix_index =  torch.argsort(torch.mm(fused_emb, fused_emb.t())[1:maxindex_uu+1, maxindex_uu+1:],dim=-1, descending=True) # [user_num*poi_num] 10*24 从小到大，indx0最小
    score_matrix_index = score_matrix_index.cpu().numpy()

    # 读取测试数据 dict, key=user, value=poi_list
    test_user_history = read_history(args.graph_path + 'test_checkin_file.txt', '\t')
    # 推荐任务预处理
    train_user_dict, test_user_dict = prepare_target(train_users_history, test_user_history)


    # 推荐、计算评价指标 3554 11433 14987
    accuracy, precision, recall, ndcg, hit_ratio, MAP = evaleate(score_matrix_index, train_user_dict, test_user_dict)

    print("Saving loss data at {}".format(args.lossdata_path))
    with open(args.lossdata_path, "wb") as ldata:
        pickle.dump(lossdata, ldata)
    # sys.exit()
