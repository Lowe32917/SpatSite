import numpy as np
# from torch_scatter import scatter_sum, scatter_mean
# from graph_feature import coordinate_graphs
# 1 最近邻均值；2 最近邻CV；
# 3-5 内积特征值；6-8 协方差特征值；9-13 线性度 平面度 球形度 各向异性 特征熵（点数少对特征值影响很大）
# 14-17 径向距离的均值、标准差、偏度、峰度；18 各向异性强度；
# 19-25 径向分布函数（RDF）

# NUM_TOPOLOGICAL_FEATURES=25

def cal_topological_features(pos, context_radius):

    RAD_INTER = 3
    CUT_INTER = 4

    if pos.shape[0] < 5:
        NUM_TOPOLOGICAL_FEATURES = 18 + np.arange(4, context_radius, RAD_INTER).shape[0] + 1
        return np.zeros(NUM_TOPOLOGICAL_FEATURES)

    feature_matrix = []
    points = pos.shape[0]
    dims = pos.shape[1]

    dot_product = np.einsum('ij,kj->ik', pos, pos)
    norm_sq = np.sum(pos ** 2, axis=1)
    dist_sq = norm_sq[:, np.newaxis] + norm_sq[np.newaxis, :] - 2 * dot_product
    np.maximum(dist_sq, 0, out=dist_sq)
    all_dist = np.sqrt(dist_sq)

    # ##############################################
    # ############### 以下是图拓扑特征 ################
    # ##############################################
    # # cut_th = torch.arange(4, context_radius, CUT_INTER)
    # # cut_th = torch.cat(cut_th, [torch.tensor([context_radius], dtype=cut_th.dtype, device=cut_th.device)])
    # # [subbatch_neigh, subbatch_center] = radius_graph(pos, r=cut_th[0], batch=batch, loop=False)
    #
    # # index_list = ['graph_density', 'average_degree', 'average_weight',
    # #               'global_efficiency', 'characteristic_path_length', 'graph_diameter',
    # #               'avg_closeness', 'std_closeness',
    # #               'avg_betweenness', 'std_betweenness',
    # #               'avg_clustering', 'algebraic_connectivity']
    # # # th_list = [10,20,30]
    # # # for i in range(3):
    # # #     temp = coordinate_graphs(pos=pos, batch=batch, max_nn=max_nn, threshold=th_list[i], device=device)
    # # #     for tk in index_list:
    # # #         feature_matrix.append(temp[tk])
    # # temp = coordinate_graphs(pos=pos, batch=batch, max_nn=max_nn, threshold=40, device=device)
    # # for tk in index_list[1:]:
    # #     feature_matrix.append(temp[tk])

    ##############################################
    ############## 以下是空间分布特征 ###############
    ##############################################

    # 最近邻距离分布指标
    all_dist += (np.eye(points, dtype=np.float32) * context_radius * 2)
    min_dist = all_dist.min(0)
    ### 最近邻 - 均值
    m_dist = np.mean(min_dist)
    feature_matrix.append(m_dist)
    ### 最近邻 - 变异系数
    D_dist = np.mean(np.square(min_dist - m_dist))
    feature_matrix.append(np.sqrt(D_dist) / m_dist)


    ##############################################
    ###### 后续操作去除了中心点，降低不必要的影响 #######
    ##############################################
    # dec_flag = np.sum(pos, dim=1) != 0
    # dec_pos = pos[dec_flag,]
    # dec_batch = batch[dec_flag]
    dec_pos = pos[1:,]

    # 内积矩阵特征值
    ### 特征值
    I_mat = dec_pos.T @ dec_pos
    I_eigenvalues = np.linalg.eigvals(I_mat)
    I_eig = np.sort(I_eigenvalues)[::-1]
    feature_matrix += list(I_eig)

    # 协方差矩阵特征值
    ### 特征值
    Z_mat = np.cov(dec_pos.T)
    Z_eigenvalues = np.linalg.eigvals(Z_mat)
    Z_eig = np.sort(Z_eigenvalues)[::-1]
    feature_matrix += list(Z_eig)
    ### 线性度
    feature_matrix.append((Z_eig[2] - Z_eig[1]) / Z_eig[2])
    ### 平面度
    feature_matrix.append((Z_eig[1] - Z_eig[0]) / Z_eig[2])
    ### 球形度
    feature_matrix.append(Z_eig[0] / Z_eig[2])
    ### 各向异性
    feature_matrix.append((Z_eig[2] - Z_eig[0]) / Z_eig[2])
    ### 特征熵
    feature_matrix.append( -( Z_eig[2] * np.log(Z_eig[2]) + Z_eig[1] * np.log(Z_eig[1]) + Z_eig[0] * np.log(Z_eig[0]) ) )

    # 径向距离
    rad_dist = all_dist[0,1:]
    ### 均值
    rad_m_dist = np.mean(rad_dist)
    feature_matrix.append(rad_m_dist)
    ### 标准差
    rad_d_dist = np.mean(np.square(rad_dist - rad_m_dist))
    rad_d_dist = np.sqrt(rad_d_dist)
    feature_matrix.append(rad_d_dist)
    ### 偏度
    Ske = np.power((rad_dist - rad_m_dist) / rad_d_dist, 3)
    Ske = np.sum(Ske) * (points-1) / ((points-2) * (points-3))
    feature_matrix.append(Ske)
    ### 峰度
    Kur = np.power((rad_dist - rad_m_dist) / rad_d_dist, 4)
    Kur = np.sum(Kur) * (points-1) * (points) / ((points-2) * (points-3) * (points-4))
    Kur = Kur - (3 * np.square(points) / ((points-1) * (points-2)))
    feature_matrix.append(Kur)

    # 各向异性
    ### 强度
    unit_pos = dec_pos / rad_dist[:,None]
    vec_D = unit_pos.mean(0)
    feature_matrix.append(np.square(vec_D).sum())

    # 径向分布函数（RDF）
    rad_bar_ten = np.arange(4, context_radius, RAD_INTER)
    rad_bar_ten = np.concatenate([[0], rad_bar_ten])
    N_total = points - 1 # 去除中心点
    R3 = context_radius ** 3
    for i in range(len(rad_bar_ten)-1):
        delta_R3 = (rad_bar_ten[i+1] ** 3) - (rad_bar_ten[i] ** 3)
        N_K = ((rad_dist > rad_bar_ten[i]) & (rad_dist <= rad_bar_ten[i+1])).sum()
        RDF_i = (N_K / N_total) * (R3 / delta_R3)
        feature_matrix.append(RDF_i)
    delta_R3 = R3 - (rad_bar_ten[-1] ** 3)
    N_K = (rad_dist > rad_bar_ten[-1]).sum()
    RDF_i = (N_K / N_total) * (R3 / delta_R3)
    feature_matrix.append(RDF_i)

    feature_matrix = np.array(feature_matrix)
    feature_matrix[np.isnan(feature_matrix)] = 0

    return feature_matrix