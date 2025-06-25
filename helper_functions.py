import numpy as np
from sklearn.impute import KNNImputer


def build_rating_matrix(train_file, return_mask=False):
    ratings = np.genfromtxt(train_file, delimiter=',', skip_header=1)

    user_ids, movie_ids = np.unique(ratings[:, 0]), np.unique(ratings[:, 1])

    user_map, movie_map = {user_id: i for i, user_id in enumerate(sorted(user_ids))}, \
                          {movie_id: j for j, movie_id in enumerate(sorted(movie_ids))}

    n, d = len(user_ids), len(movie_ids)

    rating_matrix = np.full((n, d), np.nan)
    for i in range(ratings.shape[0]):
        user_id, movie_id, rating = ratings[i, :3]
        rating_matrix[user_map[user_id], movie_map[movie_id]] = rating
    impute = KNNImputer(n_neighbors=25)
    rating_matrix = impute.fit_transform(rating_matrix)

    if return_mask == True:
        rating_matrix_masked = np.ma.masked_invalid(rating_matrix)
        return rating_matrix, rating_matrix_masked.mask, user_map, movie_map
    else:
        return rating_matrix, user_map, movie_map
