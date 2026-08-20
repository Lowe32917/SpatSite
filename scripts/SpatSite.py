import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from torch_cluster import radius_graph

# NUM_TOPOLOGICAL_FEATURES = 25
RAD_INTER = 3

class MetaMLP(torch.nn.Module):
    def __init__(self, in_hs, cut_hs, out_hs, bias, dropratio):
        super(MetaMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_hs, cut_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(cut_hs),
            nn.ELU(inplace=True),
            nn.Dropout(dropratio),
            nn.Conv1d(cut_hs, out_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_hs)
        )
    def forward(self,x_out):
        return self.mlp(x_out)

class ResidueEncoder(torch.nn.Module):
    def __init__(self, lx_ind, hx_ind, tx_ind, x_hs, dropratio, bias):
        super(ResidueEncoder, self).__init__()
        self.hx_flag = (hx_ind > 0)
        encoder_cat_hs = [128, 128, 512] if self.hx_flag else [256, 256]
        self.mlp_lx = MetaMLP(lx_ind, encoder_cat_hs[0], encoder_cat_hs[0], bias, dropratio)
        self.mlp_tx = MetaMLP(tx_ind, encoder_cat_hs[1], encoder_cat_hs[1], bias, dropratio)
        if self.hx_flag:
            self.mlp_hx = MetaMLP(hx_ind, encoder_cat_hs[2], encoder_cat_hs[2], bias, dropratio)
        self.mlp_x = MetaMLP(sum(encoder_cat_hs), x_hs, x_hs, bias, dropratio)

    def forward(self, lx, hx, tx):
        lx_out = self.mlp_lx(lx.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        tx_out = self.mlp_tx(tx.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        x_out = torch.cat([lx_out, tx_out], dim=1)
        if self.hx_flag:
            hx_out = self.mlp_hx(hx.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
            x_out = torch.cat([x_out, hx_out], dim=1)
        x_out = self.mlp_x(x_out.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        return x_out

class MetaEncoder(torch.nn.Module):
    def __init__(self, u_hs, x_hs, e_hs, tx_ind, dropratio, bias):
        super(MetaEncoder, self).__init__()
        self.mlp_e1 = MetaMLP(3, e_hs, e_hs, bias, dropratio)
        self.mlp_e2 = MetaMLP(e_hs + x_hs + x_hs, e_hs, e_hs, bias, dropratio)
        self.mlp_x1 = MetaMLP(e_hs * 2 + x_hs, x_hs, x_hs, bias, dropratio)
        self.mlp_x2 = MetaMLP(x_hs * 2, x_hs, x_hs, bias, dropratio)
        self.mlp_u = MetaMLP(x_hs * 2 + tx_ind, u_hs, u_hs, bias, dropratio)

    def forward(self, e, x, tx_center, edge_weight, edge_v, res_weight, batch):
        # 编码边特征
        e_out = self.mlp_e1(e.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        e_out = torch.cat([e_out, x[edge_v[0],], x[edge_v[1],]], dim=1)
        e_out = self.mlp_e2(e_out.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        # 编码点特征
        x1 = scatter_mean(e_out, edge_v[1].unsqueeze(-1), dim=0)
        x2 = scatter_sum(e_out * edge_weight.unsqueeze(1), edge_v[1].unsqueeze(-1), dim=0)
        x_out = torch.cat([x, x1, x2], dim=1)
        x_out = self.mlp_x1(x_out.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        x_out = torch.cat([x, x_out], dim=1)
        x_out = self.mlp_x2(x_out.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        # 编码图特征
        u1 = scatter_mean(x_out, batch.unsqueeze(-1), dim=0)
        u2 = scatter_sum(x_out * res_weight.unsqueeze(1), batch.unsqueeze(-1), dim=0)
        u_out = torch.cat([u1, u2, tx_center], dim=1)
        u_out = self.mlp_u(u_out.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        return e_out, x_out, u_out

class EdgeModel(torch.nn.Module):
    def __init__(self, e_hs, x_hs, u_hs, dropratio, bias):
        super(EdgeModel, self).__init__()
        temp_hs = e_hs + x_hs * 2 + u_hs
        self.mlp_e1 = MetaMLP(temp_hs, temp_hs // 2, e_hs, bias, dropratio)
        self.mlp_e2 = MetaMLP(e_hs * 2, e_hs, e_hs, bias, dropratio)
    def forward(self, e, x, u, edge_v, edge_batch):
        e_out = torch.cat([e, u[edge_batch], x[edge_v[0],], x[edge_v[1],]], dim=1)
        e_out = self.mlp_e1(e_out.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        e_out = torch.cat([e, e_out], dim=1)
        e_out = self.mlp_e2(e_out.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        return e_out

class NodeModel(torch.nn.Module):
    def __init__(self, e_hs, x_hs, u_hs, dropratio, bias):
        super(NodeModel, self).__init__()
        temp_hs = e_hs * 2 + x_hs + u_hs
        self.mlp_x1 = MetaMLP(temp_hs, temp_hs // 2, x_hs, bias, dropratio)
        self.mlp_x2 = MetaMLP(x_hs * 2, x_hs, x_hs, bias, dropratio)
    def forward(self, e, x, u, edge_weight, edge_v, batch):
        x1 = scatter_mean(e, edge_v[1].unsqueeze(-1), dim=0)
        x2 = scatter_sum(e * edge_weight.unsqueeze(1), edge_v[1].unsqueeze(-1), dim=0)
        x_out = torch.cat([x, x1, x2, u[batch,]], dim=1)
        x_out = self.mlp_x1(x_out.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        x_out = torch.cat([x, x_out], dim=1)
        x_out = self.mlp_x2(x_out.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        return x_out

class GraphModel(torch.nn.Module):
    def __init__(self, x_hs, u_hs, tx_ind, dropratio, bias):
        super(GraphModel, self).__init__()
        temp_hs = x_hs * 2 + u_hs + tx_ind
        self.mlp_u1 = MetaMLP(temp_hs, temp_hs // 2, u_hs, bias, dropratio)
        self.mlp_u2 = MetaMLP(u_hs * 2, u_hs, u_hs, bias, dropratio)
    def forward(self, x, u, tx_center, res_weight, batch):
        u1 = scatter_mean(x, batch.unsqueeze(-1), dim=0)
        u2 = scatter_sum(x * res_weight.unsqueeze(1), batch.unsqueeze(-1), dim=0)
        u_out = torch.cat([u, u1, u2, tx_center], dim=1)
        u_out = self.mlp_u1(u_out.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        u_out = torch.cat([u, u_out], dim=1)
        u_out = self.mlp_u2(u_out.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        return u_out

class MetaUpdater(torch.nn.Module):
    def __init__(self, u_hs, x_hs, e_hs, tx_ind, dropratio, bias, num_update):
        super(MetaUpdater, self).__init__()
        self.update_steps = num_update
        self.edge_model = nn.ModuleList(
            [EdgeModel(e_hs=e_hs, x_hs=x_hs, u_hs=u_hs, dropratio=dropratio, bias=bias)] * num_update)
        self.node_model = nn.ModuleList(
            [NodeModel(e_hs=e_hs, x_hs=x_hs, u_hs=u_hs, dropratio=dropratio, bias=bias)] * num_update)
        self.graph_model = nn.ModuleList(
            [GraphModel(x_hs=x_hs, u_hs=u_hs, tx_ind=tx_ind, dropratio=dropratio, bias=bias)] * num_update)

    def forward(self, e_out, x_out, u_out, tx_center, edge_weight, edge_v, edge_batch, res_weight, batch, center_res):
        x_list = []
        u_list = []
        for i in range(self.update_steps):
            e_out = self.edge_model[i](e=e_out, x=x_out, u=u_out, edge_v=edge_v, edge_batch=edge_batch)
            x_out = self.node_model[i](e=e_out, x=x_out, u=u_out, edge_weight=edge_weight, edge_v=edge_v, batch=batch)
            u_out = self.graph_model[i](x=x_out, u=u_out, tx_center=tx_center, res_weight=res_weight, batch=batch)
            x_list.append(x_out[center_res,])
            u_list.append(u_out)
        return x_list, u_list

class SpatSite_Model(torch.nn.Module):
    def __init__(self, lx_ind, hx_ind=0, x_hs=128, u_hs=128, t_hs=128, e_hs=128,
                 dropratio=0.5, bias=True, num_update=1, context_radius=20, max_nn=40
                 ):
        super(SpatSite_Model, self).__init__()
        self.context_radius = context_radius
        self.edge_th = context_radius / 2
        self.max_nn = max_nn
        self.num_update = num_update
        self.hx_flag = (hx_ind > 0)
        tx_ind = 18 + np.arange(4, context_radius, RAD_INTER).shape[0] + 1
        print('Model parameter: ex_flag = {}, lx_ind = {}, ex_ind = {}, tx_ind = {}'
              .format(self.hx_flag, lx_ind, hx_ind, tx_ind))
        # bn层
        self.bn = nn.BatchNorm1d(lx_ind)
        if self.hx_flag:
            self.bn_hx = nn.BatchNorm1d(hx_ind + 1)
        self.bn_tx = nn.BatchNorm1d(tx_ind)
        self.bn_ex = nn.BatchNorm1d(3)
        # 特征整合层
        self.V_encoder = ResidueEncoder(lx_ind, hx_ind + 1, tx_ind, x_hs, dropratio, bias)
        # 生成边、点、图特征
        self.encoder = MetaEncoder(u_hs, x_hs, e_hs, tx_ind, dropratio, bias)
        self.updater = MetaUpdater(u_hs, x_hs, e_hs, tx_ind, dropratio, bias, num_update)
        # 分类器
        clf_hs = (x_hs + u_hs) * num_update + tx_ind
        self.clf = nn.Sequential(
            nn.Linear(clf_hs, clf_hs // 2),
            nn.BatchNorm1d(clf_hs // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropratio),
            nn.Linear(clf_hs // 2, 2))
        print("Constructed Model.")

        print("Constructed Model.")

    def forward(self, data, device):
        pos = data.pos
        batch = data.batch
        lx = data.x
        hx = data.ex if self.hx_flag else 0
        tx = data.tx

        pos1 = pos.cpu()
        batch1 = batch.cpu()
        batch_size = torch.unique(batch).shape[0]
        # 中心点标记
        center_res = (torch.sum(torch.abs(pos), dim=1) == 0)
        if torch.sum(center_res) > (batch[-1] + 1):
            print("center_res is ERROR!!!")
        # 距离向量 和 点权重
        dist_vector = torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1)
        dist_vector[center_res] = 1
        res_weight = 1 - (torch.sqrt(torch.sum(pos * pos, dim=1)) / self.context_radius)
        res_weight[center_res] = 1
        temp = scatter_sum(res_weight, batch)
        res_weight = res_weight / temp[batch]
        # 找边 并计算边初始特征 计算边权重
        [edge_end, edge_start] = radius_graph(pos1, r=self.edge_th, batch=batch1, loop=True,
                                                         max_num_neighbors=self.max_nn).to(lx.device)
        ex = []
        ex.append(torch.sqrt(torch.sum(pos[edge_end,] * pos[edge_end,], dim=1)))
        ex.append(torch.sqrt(torch.sum(pos[edge_start,] * pos[edge_start,], dim=1)))
        temp = torch.sum(pos[edge_start,] * pos[edge_end,], dim=1) / ex[0] / ex[1]
        temp[torch.isnan(temp)] = 1
        ex.append(temp)
        ex[0] = ex[0] / self.context_radius
        ex[1] = ex[1] / self.context_radius
        ex[2] = ex[2] / 2 + 0.5
        ex = torch.stack(ex, dim=1)
        temp = pos[edge_start,] - pos[edge_end,]
        edge_weight = 1 - (torch.sqrt(torch.sum(temp * temp, dim=1)) / self.edge_th)
        temp = scatter_sum(edge_weight, edge_start)
        edge_weight = edge_weight / temp[edge_start]

        # bn层
        if self.hx_flag:
            hx = torch.cat([hx, 1 / dist_vector], dim=-1)
            hx = self.bn_hx(hx.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        lx = self.bn(lx.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        tx = self.bn_tx(tx.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        e = self.bn_ex(ex.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        # 残基特征整合
        x = self.V_encoder(lx, hx, tx)
        # 编码层
        e, x, u = self.encoder(e, x, tx_center = tx[center_res],
                               edge_weight = edge_weight, edge_v=[edge_end, edge_start],
                               res_weight=res_weight, batch=batch)
        # 更新层
        x_list, u_list = self.updater(e_out=e, x_out=x, u_out=u, tx_center = tx[center_res],
                                      edge_weight=edge_weight, edge_v=[edge_end, edge_start], edge_batch=batch[edge_start],
                                      res_weight=res_weight, batch=batch, center_res=center_res)

        output = [tx[center_res,]]
        for i in range(self.num_update):
            output.append(x_list[i])
            output.append(u_list[i])
        output = torch.cat(output, dim=1)
        score = self.clf(output)
        score = F.softmax(score, -1)
        return (score[:, 1])









