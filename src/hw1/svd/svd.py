import pandas as pd
import numpy as np
import scipy.sparse as sp
from operator import itemgetter

from src.hw1.resources import DATA_PATH
from src.hw1.logging import LOGGING_PATH

import time


class SVD:

    def __init__(self, df: pd.DataFrame, hidden_dim: int = 16):
        # id mapping
        self.movie_id_mapping = None
        self.reversed_movie_id_mapping = None
        self.user_id_mapping = None
        self.reversed_user_id_mapping = None

        self.unique_user_count = df.user_id.unique().shape[0]
        self.unique_movie_count = df.movie_id.unique().shape[0]

        self.user_item_matrix = self.get_csr_matrix(df)

        self.item_matrix = np.random.uniform(0, 1, (hidden_dim, self.unique_movie_count))
        self.user_matrix = np.random.uniform(0, 1, (self.unique_user_count, hidden_dim))

        # overall average rating parameter for regularization
        self.mu = 0 #df.rating.mean()
        self.user_bias = np.zeros(self.unique_user_count, dtype=np.float32)
        self.item_bias = np.zeros(self.unique_movie_count, dtype=np.float32)

    def get_csr_matrix(self, df: pd.DataFrame):
        """
        Function takes dataframe with column "user_id" and "movie_id" and created csr matrix filled with ratings.
        :param df: pd.DataFrame with at least 2 columns to construct csr matrix
        :return: None
        """
        ratings_df = df.sort_values('user_id')

        # mapping for movie ids so then real movie_id == its column index in user_item matrix
        self.movie_id_mapping = dict(zip(sorted(ratings_df.movie_id.unique()), np.arange(self.unique_movie_count)))
        self.reversed_movie_id_mapping = dict(
            zip(np.arange(self.unique_movie_count), sorted(ratings_df.movie_id.unique())))

        # mapping for user ids, real user_id == its row index in user_item matrix
        self.user_id_mapping = dict(zip(sorted(ratings_df.user_id.unique()), np.arange(self.unique_user_count)))
        self.reversed_user_id_mapping = dict(
            zip(np.arange(self.unique_user_count), sorted(ratings_df.user_id.unique())))
        self.user_item_matrix = sp.csr_matrix((ratings_df.rating,
                                               (itemgetter(*ratings_df.user_id)(self.user_id_mapping),
                                                itemgetter(*ratings_df.movie_id)(self.movie_id_mapping))),
                                              shape=(self.unique_user_count, self.unique_movie_count))

        return self.user_item_matrix

    def loss(self, scores: np.array, predictions: np.array, user_bias: np.array, item_bias: np.array) -> np.ndarray:
        mse = np.square(scores - (predictions + user_bias + item_bias + self.mu))
        return mse

    def fit(self, eps: float = 1e-1, max_iter: int = 1_000, n_samples_to_optimize: int = 500, lr: float = 1e-3,
            gamma: float = 0.001):
        mse = np.inf
        current_iter = 0
        print('Start fitting the model...')
        start_time = time.time()
        loss_logger = []

        while current_iter < max_iter and np.mean(mse) > eps:
            current_iter += 1
            i = np.random.choice(list(self.user_id_mapping.values()), n_samples_to_optimize)
            j = np.random.choice(list(self.movie_id_mapping.values()), n_samples_to_optimize)

            target_scores = self.user_item_matrix.toarray()[i, j]
            predicted_scores = np.einsum('ij, ij -> i', self.user_matrix[i, :], self.item_matrix[:, j].T)

            user_vector_regularization = np.square(np.linalg.norm(self.user_matrix[i, :], axis=1))
            item_vector_regularization = np.square(np.linalg.norm(self.item_matrix[:, j], axis=0))
            regularization = gamma * (
                    np.sqrt(self.user_bias[i]) + np.sqrt(self.item_bias[j]) + user_vector_regularization
                    + item_vector_regularization
            )
            mse = self.loss(target_scores, predicted_scores, self.user_bias[i], self.item_bias[j]) + regularization
            loss_logger.append(np.mean(mse))

            # update weights
            self.user_bias[i] += np.clip(lr * (mse - (gamma * self.user_bias[i])), a_min=1e-8, a_max=1e-1)
            self.item_bias[j] += np.clip(lr * (mse - (gamma * self.item_bias[j])), a_min=1e-8, a_max=1e-1)

            self.user_matrix[i, :] -= lr * (np.einsum(
                'i, ij -> ij', mse, self.item_matrix[:, j].T) - (gamma * self.user_matrix[i, :]))
            self.item_matrix[:, j] -= lr * (np.einsum(
                'i, ij -> ij', mse, self.user_matrix[i, :]).T - (gamma * self.item_matrix[:, j]))

        # logging loss value
        loss_logger_df = pd.DataFrame({'iter': range(len(loss_logger)), 'mse': loss_logger})
        loss_logger_df.to_csv(f"{LOGGING_PATH.parent}/lr{lr}_max_iter{max_iter}_hidden_dim{self.user_matrix.shape[1]}"
                              f"_samples{n_samples_to_optimize}",
                              index=False)
        print(f'Model fitted in {int(time.time() - start_time)} seconds.')
        return self.user_matrix, self.item_matrix


# ratings = pd.read_csv(f'{DATA_PATH.parent}/ratings.dat', delimiter='::', header=None,
#                       names=['user_id', 'movie_id', 'rating', 'timestamp'],
#                       usecols=['user_id', 'movie_id', 'rating'], engine='python')
#
# svd_model = SVD(ratings)
# user_matrix, item_matrix = svd_model.fit()
