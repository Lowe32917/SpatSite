import numpy as np
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


    all_dist += (np.eye(points, dtype=np.float32) * context_radius * 2)
    min_dist = all_dist.min(0)
    m_dist = np.mean(min_dist)
    feature_matrix.append(m_dist)
    D_dist = np.mean(np.square(min_dist - m_dist))
    feature_matrix.append(np.sqrt(D_dist) / m_dist)

    dec_pos = pos[1:,]

    I_mat = dec_pos.T @ dec_pos
    I_eigenvalues = np.linalg.eigvals(I_mat)
    I_eig = np.sort(I_eigenvalues)[::-1]
    feature_matrix += list(I_eig)

    Z_mat = np.cov(dec_pos.T)
    Z_eigenvalues = np.linalg.eigvals(Z_mat)
    Z_eig = np.sort(Z_eigenvalues)[::-1]
    feature_matrix += list(Z_eig)
    Linearity_temp = (Z_eig[0] - Z_eig[1]) / Z_eig[0]
    feature_matrix.append(Linearity_temp)
    Planarity = (Z_eig[1] - Z_eig[2]) / Z_eig[0]
    feature_matrix.append(Planarity)
    Sphericity_temp = Z_eig[2] / Z_eig[0]
    feature_matrix.append(Sphericity_temp)
    Anisotropy_temp = (Z_eig[0] - Z_eig[2]) / Z_eig[0]
    feature_matrix.append(Anisotropy_temp)
    nor_Z_eig = Z_eig / Z_eig.sum()
    eigenentropy_temp =  -( nor_Z_eig[2] * np.log(nor_Z_eig[2]) +
                            nor_Z_eig[1] * np.log(nor_Z_eig[1]) +
                            nor_Z_eig[0] * np.log(nor_Z_eig[0]) )
    feature_matrix.append(eigenentropy_temp / np.log(3))

    rad_dist = all_dist[0,1:]
    rad_m_dist = np.mean(rad_dist)
    feature_matrix.append(rad_m_dist)
    rad_d_dist = np.mean(np.square(rad_dist - rad_m_dist))
    rad_d_dist = np.sqrt(rad_d_dist)
    feature_matrix.append(rad_d_dist)
    Ske = np.power((rad_dist - rad_m_dist) / rad_d_dist, 3)
    Ske = np.sum(Ske) * (points-1) / ((points-2) * (points-3))
    feature_matrix.append(Ske)
    Kur = np.power((rad_dist - rad_m_dist) / rad_d_dist, 4)
    Kur = np.sum(Kur) * (points-1) * (points) / ((points-2) * (points-3) * (points-4))
    Kur = Kur - (3 * np.square(points) / ((points-1) * (points-2)))
    feature_matrix.append(Kur)


    unit_pos = dec_pos / rad_dist[:,None]
    vec_D = unit_pos.mean(0)
    feature_matrix.append(np.square(vec_D).sum())

    rad_bar_ten = np.arange(4, context_radius, RAD_INTER)
    rad_bar_ten = np.concatenate([[0], rad_bar_ten])
    N_total = points - 1
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