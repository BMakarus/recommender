import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.model_selection import KFold
from modules.helper_functions import build_rating_matrix
from matplotlib import pyplot as plt
import torch


def train_nmf_model(train_file, n_components=5):
    Z, user_map, movie_map = build_rating_matrix(train_file)

    model = NMF(n_components=n_components, init='random', random_state=0)
    W = model.fit_transform(Z)
    H = model.components_
    Z_approx = np.dot(W, H)

    return Z_approx, user_map, movie_map


def train_svd_model(train_file, n_components=5):
    Z, user_map, movie_map = build_rating_matrix(train_file)

    model = TruncatedSVD(n_components=n_components, random_state=0)
    model.fit(Z)
    W = model.transform(Z) / model.singular_values_
    Lambda = np.diag(model.singular_values_)
    V_T = model.components_
    H = np.dot(Lambda, V_T)
    Z_approx = np.dot(W, H)

    return Z_approx, user_map, movie_map


def train_svd2_model(train_file, n_components=5, n_iter=10):
    Z, mask, user_map, movie_map = build_rating_matrix(train_file, return_mask=True)

    for i in range(n_iter):
        model = TruncatedSVD(n_components=n_components, random_state=0)
        model.fit(Z)
        W = model.transform(Z) / model.singular_values_
        Lambda = np.diag(model.singular_values_)
        V_T = model.components_
        H = np.dot(Lambda, V_T)
        Z_approx = np.dot(W, H)
        Z[np.where(mask == True)] = Z_approx[np.where(mask == True)]
    return Z, user_map, movie_map


def train_sgd_model(train_file, n_components=5, lr=0.0001, n_epochs=1000, batch_size=20000):
    ratings = np.genfromtxt(train_file, delimiter=',', skip_header=1)

    user_ids, movie_ids = np.unique(ratings[:, 0]), np.unique(ratings[:, 1])

    user_map, movie_map = {user_id: i for i, user_id in enumerate(sorted(user_ids))}, \
                          {movie_id: j for j, movie_id in enumerate(sorted(movie_ids))}

    n, d = len(user_ids), len(movie_ids)

    Z = np.full((n, d), np.nan)
    for i in range(ratings.shape[0]):
        user_id, movie_id, rating = ratings[i, :3]
        Z[user_map[user_id], movie_map[movie_id]] = rating
    Z = np.ma.masked_invalid(Z)
    Z_mask_torch = torch.from_numpy(Z.mask)
    Z_torch = torch.from_numpy(Z.data)

    dataloader = torch.utils.data.DataLoader(Z_torch, batch_size=batch_size, shuffle=True)

    W = torch.randn((n, n_components), requires_grad=True, dtype=torch.float, device="cpu")
    H = torch.randn((n_components, d), requires_grad=True, dtype=torch.float, device="cpu")

    optimizer = torch.optim.SGD([W, H], lr=lr)

    for epoch in range(n_epochs):
            for batch in dataloader:
                loss = torch.sum((Z_torch[torch.where(Z_mask_torch == False)] - torch.matmul(torch.exp(W), torch.exp(H))[
                    torch.where(Z_mask_torch == False)]) ** 2)+10*torch.sum(torch.exp(W))+10*torch.sum(torch.exp(H))


                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

    return torch.exp(W), torch.exp(H), user_map, movie_map


