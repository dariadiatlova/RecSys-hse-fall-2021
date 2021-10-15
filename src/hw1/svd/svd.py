import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_distances
import time


class SVD:
    def __init__(self, user_item_matrix: sp.csr_matrix, hidden_dim: int = 16):
        """
        Classical SVD Algorithm with fit and find top_k similar methods.
        :param user_item_matrix: sp.csr.csr_matrix of shape n_users X n_movies.
        :param hidden_dim: int, hidden size of space where user and item embedding will be mapped.
        """
        self.user_item_matrix = user_item_matrix
        self.user_non_zero_idx, self.item_non_zero_idx = self.user_item_matrix.nonzero()
        self.item_matrix = np.random.uniform(0, 1 / np.sqrt(hidden_dim),
                                             (hidden_dim, user_item_matrix.toarray().shape[1]))
        self.user_matrix = np.random.uniform(0, 1 / np.sqrt(hidden_dim),
                                             (user_item_matrix.toarray().shape[0], hidden_dim))

        # overall average rating parameter for regularization
        self.user_bias = np.zeros(user_item_matrix.toarray().shape[0], dtype=np.float32)
        self.item_bias = np.zeros(user_item_matrix.toarray().shape[1], dtype=np.float32)

    def _update_params(self, loss_item: float, user_idx: int, item_idx: int, update_bias: bool,
                       lr: float, gamma: float, beta: float) -> None:
        """
        Function updates user, item vector and user_bias, item_bias vectors.
        :param loss_item: float – prediction error.
        :param user_idx: id of user from user/item pair.
        :param item_idx: if of item from user/item pair.
        :param update_bias: bool: if true, learn user and bias vector.
        :param lr: float: learning rate used for updating all 4 vectors.
        :param gamma: float: hyper parameter used for updating user and item vectors.
        :param beta: float: hyper parameter used for updating user and item bias vectors.
        :return: None
        """

        self.user_matrix[user_idx, :] -= lr * (
                loss_item * self.item_matrix[:, item_idx].T + gamma * self.user_matrix[user_idx, :])
        self.item_matrix[:, item_idx] -= lr * (
                loss_item * self.user_matrix[user_idx, :].T + gamma * self.item_matrix[:, item_idx])
        if update_bias:
            self.user_bias[user_idx] -= lr * (loss_item + beta * self.user_bias[user_idx])
            self.item_bias[item_idx] -= lr * (loss_item + beta * self.item_bias[item_idx])

    def _loss(self, prediction: float, score: int, user_bias: float, item_bias: float) -> float:
        """
        Function calculates current score error on prediction stage.
        :param prediction: float: current score computed by user / item matrix multiplication.
        :param score: int: real score from given csr matrix.
        :param user_bias: float: user bias value for the given user.
        :param item_bias: float: item bias value for the given item.
        :return: float: err
        """
        loss_item = prediction + user_bias + item_bias - score
        return loss_item

    def mse(self) -> float:
        """
        Method returns mse computed on all non_zero values from the target matrix.
        :return: float: mean squared error.
        """
        current_prediction_matrix = np.dot(self.user_matrix, self.item_matrix)
        return np.mean(np.square(current_prediction_matrix[self.user_non_zero_idx, self.item_non_zero_idx]
                                 - self.user_item_matrix.toarray()[self.user_non_zero_idx, self.item_non_zero_idx]))

    def fit(self, lr: float = 0.01, gamma: float = 0.001, beta: float = 100., epochs: int = 20,
            update_bias: bool = True, num_samples_to_update: int = None, logging_path: str = None):
        """
        Basic training loop implementation.
        :param lr: float: learning rate used for updating all 4 vectors.
        :param gamma: float: hyper parameter used for updating user and item vectors.
        :param beta: float: hyper parameter used for updating user and item bias vectors.
        :param epochs: int: total number of epochs to use for updating user/item pairs.
        :param update_bias: bool: if true, learn user and bias vector.
        :param num_samples_to_update: int, if not set – equal to all non zero values in the given matrix, so all pairs
        will bi used for learning during the epoch.
        :param logging_path: path to the folder to write csv file with mse computed on each epoch.
        :return: Tuple[np.ndarray, np.ndarray] – user and item matrices. User matrix n_users x hidden_dim, item matrix:
        hidden_dim x n_items.
        """
        if not num_samples_to_update:
            num_samples_to_update = self.user_non_zero_idx.shape[0]

        assert num_samples_to_update <= self.user_non_zero_idx.shape[0], \
            f"Maximum value for num_samples_to_update param  is {self.user_non_zero_idx.shape[0]}."

        print('Start fitting the model...')
        start_time = time.time()
        loss_logger = []
        user_item_matrix = self.user_item_matrix.toarray()

        for epoch in range(epochs):
            # sample n pairs to update on each epoch
            pairs_idx = np.random.choice(np.arange(self.item_non_zero_idx.shape[0]), num_samples_to_update)
            for i, idx in enumerate(pairs_idx):
                user_idx, item_idx = self.user_non_zero_idx[idx], self.item_non_zero_idx[idx]
                predicted_score = np.dot(self.user_matrix[user_idx, :], self.item_matrix[:, item_idx])
                target_score = user_item_matrix[user_idx][item_idx]
                loss_item = self._loss(
                    predicted_score, target_score, self.user_bias[user_idx], self.item_bias[item_idx])
                self._update_params(loss_item, user_idx, item_idx, update_bias, lr, gamma, beta)
            print(f"Wow, {epoch}th is fitted in {int(time.time() - start_time) // 60} minutes!")
            mse = self.mse()
            loss_logger.append(mse)
            print(f"Epoch: {epoch}, mse: {mse}.")

        if logging_path:
            loss_logger_df = pd.DataFrame({'epoch': range(len(loss_logger)), 'mse': loss_logger})
            loss_logger_df.to_csv(f"{logging_path}/lr{lr}_epoch{epochs}_hidden_dim{self.user_matrix.shape[1]}"
                                  f"_samples{num_samples_to_update}", index=False)

        print(f'Model fitted in {int(time.time() - start_time) / 60} minutes.')
        return self.user_matrix, self.item_matrix

    def get_k_similar_movies(self, item_ids: np.array, n: int = 5, metric: str='cosine'):
        """
        Function take an 1d array of item_ids and return 2d with the list of the most similar movies.
        :param item_ids: np.ndarray with movie indices
        :param n: int: number of items to search among similar.
        :param metric: str: dot, otherwise cosine distance.
        :return: 2d np.ndarray of shape n_indices x n
        """
        if metric == "dot":
            predicted_scores = np.dot(self.item_matrix[:, item_ids].T, self.item_matrix)
        else:
            predicted_scores = 1 - cosine_distances(self.item_matrix[:, item_ids].T, self.item_matrix.T)
        predictions = np.argsort(-predicted_scores)[:, :n + 1]
        return predictions[:, 1:]
