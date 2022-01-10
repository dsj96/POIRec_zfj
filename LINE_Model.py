'''
Descripttion: 
version: 
Date: 2021-12-28 23:23:11
LastEditTime: 2022-01-10 22:04:23
'''
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
# 关闭主机和设备间的异步执行，获取更多的输出信息


# def forward(self, x, adj):
#     x = F.relu(self.gc1(x, adj)) # (2708, 16) = (2708,1433) (2708, 2708)
#     x = F.dropout(x, self.dropout, training=self.training)
#     x = self.gc2(x, adj)  # (2708, 7)
#     return F.log_softmax(x, dim=1) # 在维度为1的条件进行log_softmax操作
#     torch.cat((self.nodes_embeddings.weight, self.nodes_embeddings.weight),dim=0)
#     torch.LongTensor([[1,2,4,5],[4,3,2,9]]) embedding = nn.Embedding(10, 3)  embedding(input)

# 继承自nn.Module
class LINEModel(nn.Module):
    def __init__(self, size, embed_dim=128, order=1):
        super(LINEModel, self).__init__()

        assert order in [1, 2], print("Order should either be int(1) or int(2)")
        # 设置embedding的维度
        self.embed_dim = embed_dim
        # 论文中的1st order, 2nd order
        self.order = order
        # nodes数*embedding维度
        self.nodes_embeddings = nn.Embedding(size, embed_dim)

        # 初始化模型参数
        # 只有1st order时每个node只需要一个embedding
        # 当有2nd order时每个node还需要一个context embedding（邻居），共计两个
        if order == 2:
            self.contextnodes_embeddings = nn.Embedding(size, embed_dim)
            # uniform的Initialization
            self.contextnodes_embeddings.weight.data = self.contextnodes_embeddings.weight.data.uniform_(-.5, .5) / embed_dim

        # uniform的Initialization
        self.nodes_embeddings.weight.data = self.nodes_embeddings.weight.data.uniform_(-.5, .5) / embed_dim

    def forward(self, v_i, v_j, negsamples, device):

        v_i = self.nodes_embeddings(v_i)


        # 这里是1阶2阶相似度计算的区别，2阶是用上下文contextnodes_embeddings
        # 1阶用的是nodes_embeddings
        if self.order == 2:
            v_j = self.contextnodes_embeddings(v_j)
            negativenodes = -self.contextnodes_embeddings(negsamples)

        else:
            v_j = self.nodes_embeddings(v_j)
            negativenodes = -self.nodes_embeddings(negsamples)

        # 公式（7）中的第一项（正样本计算），第一步是点乘，然后是按行求和
        mulpositivebatch = torch.mul(v_i, v_j)
        positivebatch = F.logsigmoid(torch.sum(mulpositivebatch, dim=1))
        # 公式（7）中的第二项（负样本计算）
        mulnegativebatch = torch.mul(v_i.view(len(v_i), 1, self.embed_dim), negativenodes) # 5000,1,128 5000,5,128 5000,5,128
        negativebatch = torch.sum(F.logsigmoid(torch.sum(mulnegativebatch, dim=2)), dim=1)
        # 公式（7）
        loss = positivebatch + negativebatch
        return -torch.mean(loss)


class MultiViewModel(nn.Module):
    def __init__(self, size_uu, size_up, size_pp, embed_dim=128, order=1):
        super(MultiViewModel, self).__init__()

        assert order in [1, 2], print("Order should either be int(1) or int(2)")

        self.order = order

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 设置embedding的维度
        self.LINE_uu = LINEModel(size_uu, embed_dim, order)
        self.LINE_up = LINEModel(size_up, embed_dim, order)
        self.LINE_pp = LINEModel(size_pp, embed_dim, order)

        # 主要是为了从Embedding中取出向量Tensor
        self.num_nodes_uu = self.LINE_uu.nodes_embeddings.num_embeddings
        self.num_nodes_up = self.LINE_up.nodes_embeddings.num_embeddings
        # self.num_nodes_pp = self.LINE_pp.nodes_embeddings.num_embeddings
        self.num_nodes_pp = self.num_nodes_up - self.num_nodes_uu

        self.index_uu  = torch.LongTensor([i+1 for i in range(self.num_nodes_uu-1)])
        self.index_up  = torch.LongTensor([i+1 for i in range(self.num_nodes_up-1)])
        self.index_pp  = torch.LongTensor([i+1 for i in range(self.num_nodes_uu-1, self.num_nodes_up-1)])
        self.index_pad = pad_emb_idx     = torch.LongTensor([0])

        self.attention = Attention(embed_dim)


    def supervised_obj(self, fused_emb, users_history):
        # poi_list 用户去过同一个poi几次就计算几次损失
        # poi_set  去过的poi只计算1次损失
        # TODO: u_history was a list [user,[poi_list],[poi_set]]
        cur_loss = torch.tensor( 0.,dtype=torch.float32)
        for user_history in users_history:
            user_id, user_h = user_history[0], user_history[1]
            user_vec  = fused_emb[user_id:user_id+1, :]    # [1*128]
            pois_vec  = fused_emb[user_h, :]    # [n*128]
            cur_loss = cur_loss + torch.sum(torch.mm(pois_vec, user_vec.t()))
        return -cur_loss

    def forward(self, v_i_uu, v_j_uu, negsamples_uu, v_i_up, v_j_up, negsamples_up, v_i_pp, v_j_pp, negsamples_pp, users_history, device):

        loss_uu = self.LINE_uu(v_i_uu, v_j_uu, negsamples_uu, device)
        loss_up = self.LINE_up(v_i_up, v_j_up, negsamples_up, device)
        loss_pp = self.LINE_pp(v_i_pp, v_j_pp, negsamples_pp, device)

        # if self.order == 2:
        #     cur_uu_contex_embedding = self.LINE_uu.contextnodes_embeddings
        #     cur_up_contex_embedding = self.LINE_up.contextnodes_embeddings
        #     cur_pp_contex_embedding = self.LINE_pp.contextnodes_embeddings
        # 变为了Tensor
        cur_uu_embedding  = self.LINE_uu.nodes_embeddings(self.index_uu)
        cur_up_embedding  = self.LINE_up.nodes_embeddings(self.index_up)
        cur_pp_embedding  = self.LINE_pp.nodes_embeddings(self.index_pp)
        cur_pad_embedding = self.LINE_pp.nodes_embeddings(self.index_pad)


        fused_emb       = self.attention(torch.cat([cur_pad_embedding, cur_uu_embedding, cur_pp_embedding], dim=0), \
                                         torch.cat([cur_pad_embedding, cur_up_embedding], dim=0))
        loss_supervised = self.supervised_obj(fused_emb, users_history)

        # return loss_uu + loss_up + loss_pp + loss_supervised, fused_emb
        return loss_uu + loss_up + loss_pp, fused_emb



class Attention(nn.Module):
    def __init__(self, embed_dim=128):
        super(Attention, self).__init__()

        # 设置embedding的维度
        self.fc = nn.Linear(embed_dim, embed_dim, bias=True)
        self.common_vec = nn.Linear(embed_dim, 1, bias=False)


    def forward(self, A_embedding, B_embedding):

        A_embedding_att = self.common_vec(
            F.leaky_relu(
                self.fc(A_embedding)
            )
        ) # num node * 1

        B_embedding_att = self.common_vec(
            F.leaky_relu(
                self.fc(B_embedding)
            )
        ) # num node * 1

        # 行和为1
        att_score = F.softmax(torch.cat([A_embedding_att, B_embedding_att], dim=1), dim=1) # # num node * 2
        # [68*128] [68,1]
        # 参考 https://blog.csdn.net/qq_34243930/article/details/106889639?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164080163816780261967145%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=164080163816780261967145&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~baidu_landing_v2~default-2-106889639.nonecase&utm_term=pytorch+%E7%82%B9%E4%B9%98&spm=1018.2226.3001.4450
        fused_embedding = torch.mul(A_embedding, att_score[:,0].unsqueeze(-1)) + torch.mul(B_embedding, att_score[:,1].unsqueeze(-1))


        # TODO: 接着fused_embedding计算OBJ3+数据预处理换id
        return fused_embedding