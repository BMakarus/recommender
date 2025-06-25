import numpy as np
import torch
from modules.train_functions import train_nmf_model, train_svd_model, train_svd2_model, train_sgd_model
from matplotlib import pyplot as plt

def predict_nmf(test_file, model_data):
    ratings = np.genfromtxt(test_file, delimiter=',', skip_header=1)

    Z_approx, user_map, movie_map = model_data["Z_approx"], model_data["user_map"], model_data["movie_map"]

    predictions = []
    for i in range(ratings.shape[0]):
        user_id, movie_id = ratings[i, :2]
        if user_id in user_map and movie_id in movie_map:
            i, j = user_map[user_id], movie_map[movie_id]
            rating = Z_approx[i, j]
        elif user_id in user_map:
            i = user_map[user_id]
            rating = np.mean(Z_approx[i, :])
        elif movie_id in movie_map:
            j = movie_map[movie_id]
            rating = np.mean(Z_approx[:, j])
        else:
            rating = 0
        rating_rounded = round(rating*2)/2
        predictions.append({"userId": user_id, "movieId": movie_id, "rating": rating_rounded})
    return predictions


def predict_svd(test_file, model_data):
    ratings = np.genfromtxt(test_file, delimiter=',', skip_header=1)

    Z_approx, user_map, movie_map = model_data["Z_approx"], model_data["user_map"], model_data["movie_map"]

    predictions = []
    for i in range(ratings.shape[0]):
        user_id, movie_id = ratings[i, :2]
        if user_id in user_map and movie_id in movie_map:
            i, j = user_map[user_id], movie_map[movie_id]
            rating = Z_approx[i, j]
        elif user_id in user_map:
            i = user_map[user_id]
            rating = np.mean(Z_approx[i, :])
        elif movie_id in movie_map:
            j = movie_map[movie_id]
            rating = np.mean(Z_approx[:, j])
        else:
            rating = 0
        rating_rounded = round(rating*2)/2
        predictions.append({"userId": user_id, "movieId": movie_id, "rating": rating_rounded})
    return predictions


def predict_svd2(test_file, model_data):
    ratings = np.genfromtxt(test_file, delimiter=',', skip_header=1)

    Z_approx, user_map, movie_map = model_data["Z_approx"], model_data["user_map"], model_data["movie_map"]

    predictions = []
    for i in range(ratings.shape[0]):
        user_id, movie_id = ratings[i, :2]
        if user_id in user_map and movie_id in movie_map:
            i, j = user_map[user_id], movie_map[movie_id]
            rating = Z_approx[i, j]
        elif user_id in user_map:
            i = user_map[user_id]
            rating = np.mean(Z_approx[i, :])
        elif movie_id in movie_map:
            j = movie_map[movie_id]
            rating = np.mean(Z_approx[:, j])
        else:
            rating = 0
        rating_rounded = round(rating*2)/2
        predictions.append({"userId": user_id, "movieId": movie_id, "rating": rating_rounded})
    return predictions


def predict_sgd(test_file, model_data):
    ratings = np.genfromtxt(test_file, delimiter=',', skip_header=1)

    W, H, user_map, movie_map = model_data["W"], model_data["H"], model_data["user_map"], model_data["movie_map"]
    Z_approx = torch.matmul(W, H).numpy()

    predictions = []
    for i in range(ratings.shape[0]):
        user_id, movie_id = ratings[i, :2]
        if user_id in user_map and movie_id in movie_map:
            i, j = user_map[user_id], movie_map[movie_id]
            rating = Z_approx[i, j]
        elif user_id in user_map:
            i = user_map[user_id]
            rating = np.mean(Z_approx[i, :])
        elif movie_id in movie_map:
            j = movie_map[movie_id]
            rating = np.mean(Z_approx[:, j])
        else:
            rating = 0
        rating_rounded = round(rating*2)/2
        predictions.append({"userId": user_id, "movieId": movie_id, "rating": rating_rounded})
    return predictions


def predict_all(test_file, models_data):
    ratings = np.genfromtxt(test_file, delimiter=',', skip_header=1)

    Z_approx_nmf, user_map, movie_map = models_data[0]
    Z_approx_svd = models_data[1][0]
    Z_approx_svd2 = models_data[2][0]
    W, H = model_data[3]["W"], model_data[3]["H"]
    Z_approx_sgd = torch.matmul(W, H).numpy()

    predictions = []
    for i in range(ratings.shape[0]):
        user_id, movie_id = ratings[i, :2]
        if user_id in user_map and movie_id in movie_map:
            i, j = user_map[user_id], movie_map[movie_id]
            rating_nmf = Z_approx_nmf[i, j]
            rating_svd = Z_approx_svd[i, j]
            rating_svd2 = Z_approx_svd2[i, j]
            rating_sgd = Z_approx_sgd[i, j]
        elif user_id in user_map:
            i = user_map[user_id]
            rating_nmf = np.mean(Z_approx_nmf[i, :])
            rating_svd = np.mean(Z_approx_svd[i, :])
            rating_svd2 = np.mean(Z_approx_svd2[i, :])
            rating_sgd = np.mean(Z_approx_sgd[i, :])
        elif movie_id in movie_map:
            j = movie_map[movie_id]
            rating_nmf = np.mean(Z_approx_nmf[:, j])
            rating_svd = np.mean(Z_approx_svd[:, j])
            rating_svd2 = np.mean(Z_approx_svd2[:, j])
            rating_sgd = np.mean(Z_approx_sgd[:, j])
        else:
            rating_nmf, rating_svd, rating_svd2, rating_sgd = 0, 0, 0, 0
        rating_nmf_rounded = round(rating_nmf*2)/2
        rating_svd_rounded = round(rating_svd * 2) / 2
        rating_svd2_rounded = round(rating_svd2 * 2) / 2
        rating_sgd_rounded = round(rating_sgd * 2) / 2
        predictions.append({"userId": user_id, "movieId": movie_id, "rating_nmf": rating_nmf_rounded,
                            "rating_svd": rating_svd_rounded, "rating_svd2": rating_svd2_rounded,
                            "rating_sgd": rating_sgd_rounded})
    return predictions

