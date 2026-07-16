import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import faiss
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
import torch.nn as nn

class GCFMCL(nn.Module):

    def __init__(self, args, dataset):
        super(GCFMCL, self).__init__()

        self.n_drug = args.n_Drug
        self.n_ncrna = args.n_NcRNA

        self.device = args.device

        self.dataset = dataset

        # load parameters info
        self.latent_dim = args.embedding_size  # int type: the embedding size of the base model
        self.n_layers = args.n_layers  # int type: the layer num of the base model
        self.reg_weight = args.reg_weight  # float32 type: the weight decay for l2 normalization

        self.ssl_temp = args.ssl_temp  #
        self.ssl_reg = args.ssl_reg
        self.hyper_layers = args.hyper_layers
        self.temperature = 0.1
        self.alpha = args.alpha

        self.proto_reg = args.proto_reg
        self.k = args.num_clusters

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_drug, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_ncrna, embedding_dim=self.latent_dim)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.norm_adj_mat = self.get_norm_adj_mat(self.dataset.pos_edge_label_index).to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

    def e_step(self):
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        item_embeddings = self.item_embedding.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """
        Run K-means algorithm to get k clusters of the input tensor x
        """
        kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def get_norm_adj_mat(self, edge_index=None):

        if edge_index is not None:

            edges = edge_index.numpy()

            user_ids = edges[0]
            item_ids = edges[1]

            from scipy.sparse import coo_matrix
            inter_M = coo_matrix(
                (np.ones(len(user_ids)), (user_ids, item_ids-self.n_drug)),
                shape=(self.n_drug, self.n_ncrna)
            ).astype(np.float32)
        else:
            print("edge_index is None")

        A = sp.dok_matrix((self.n_drug + self.n_ncrna, self.n_drug + self.n_ncrna), dtype=np.float32)

        inter_M_t = inter_M.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_drug), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_drug, inter_M_t.col), [1] * inter_M_t.nnz)))
        A.update(data_dict)

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)

        D = sp.diags(diag)
        L = D @ A @ D

        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        return SparseL

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def get_embeddings(self):
        u_emb = self.user_embedding
        i_emb = self.item_embedding
        u_wei = self.user_embedding.weight
        i_wei = self.item_embedding.weight
        return u_emb, i_emb, u_wei, i_wei

    def forward(self):
        embeddings = self.get_ego_embeddings()
        embeddings_list = [embeddings]
        for layer_idx in range(max(self.n_layers, self.hyper_layers * 2)):
            if layer_idx == 0:
                all_embeddings = torch.sparse.mm(self.norm_adj_mat, embeddings)
            else:
                all_embeddings = torch.sparse.mm(self.norm_adj_mat, all_embeddings)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers + 1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_drug, self.n_ncrna])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def ProtoNCE_loss(self, node_embedding, user, item):
        user_embeddings_all, item_embeddings_all = torch.split(node_embedding, [self.n_drug, self.n_ncrna])

        user_embeddings = user_embeddings_all[user]  # [B, e]
        norm_user_embeddings = F.normalize(user_embeddings)

        user2cluster = self.user_2cluster[user]  # [B,]
        user2centroids = self.user_centroids[user2cluster]  # [B, e]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.item_2cluster[item]  # [B, ]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_drug, self.n_ncrna])
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding,
                                                                                 [self.n_drug, self.n_ncrna])

        current_user_embeddings = current_user_embeddings[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_item_embeddings[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def infoNCE_Loss(self, lastlayer_emb, firstlayer_emb, pos_index, neg_index, temperature=0.1):

        posemb_last = F.normalize(lastlayer_emb[pos_index])  # [num_pos, dim]
        posemb_first = F.normalize(firstlayer_emb[pos_index])  # [num_pos, dim]
        negemb_first = F.normalize(firstlayer_emb[neg_index])  # [num_neg, dim]

        pos_sim = torch.mul(posemb_last, posemb_first).sum(dim=-1)  # [num_pos]

        neg_sim = torch.matmul(posemb_last, negemb_first.T)  # [num_pos, num_neg]

        pos_exp = torch.exp(pos_sim / temperature)  # [num_pos]

        neg_exp = torch.exp(neg_sim / temperature)  # [num_pos, num_neg]
        denom = pos_exp.unsqueeze(1) + neg_exp.sum(dim=1, keepdim=True)  # [num_pos, 1]

        loss = -torch.log(pos_exp.unsqueeze(1) / denom).mean()

        return loss


    def calculate_loss(self, train_data):

        posINDEX_drug = []
        posINDEX_mirna = []
        negINDEX_drug = []
        negINDEX_mirna = []

        for d, m in zip(train_data.neg_edge_label_index[0], train_data.neg_edge_label_index[1]):
            negINDEX_drug.append(int(d))
            negINDEX_mirna.append(int(m))

        for d, m in zip(train_data.pos_edge_label_index[0], train_data.pos_edge_label_index[1]):
            posINDEX_drug.append(int(d))
            posINDEX_mirna.append(int(m))

        drug_all_embeddings, mirna_all_embeddings, embeddings_list = self.forward()

        info_nce_LossD = self.infoNCE_Loss(
            embeddings_list[-1][:],
            embeddings_list[0][:],
            posINDEX_drug,
            negINDEX_drug,
            self.temperature
        )

        info_nce_LossM = self.infoNCE_Loss(
            embeddings_list[-1][:],
            embeddings_list[0][:],
            posINDEX_mirna,
            negINDEX_mirna,
            self.temperature
        )


        total_loss = (info_nce_LossD +
                      info_nce_LossM )

        return total_loss

    def predict(self, interaction):
        user = range(self.n_drug)
        item = range(self.n_ncrna)

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        u_embeddings = user_all_embeddings[user].cpu().detach().numpy().tolist()
        i_embeddings = item_all_embeddings[item].cpu().detach().numpy().tolist()

        drug = []
        rna = []
        for i in range(len(u_embeddings)):
            if u_embeddings[i] not in drug:
                drug.append(u_embeddings[i])
        for i in range(len(i_embeddings)):
            if i_embeddings[i] not in rna:
                rna.append(i_embeddings[i])
        return drug, rna
