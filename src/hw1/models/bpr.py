from src.hw1.util import MatrixFactorization
from collections import defaultdict
import numpy as np
import time
from scipy.special import expit


class BPR(MatrixFactorization):
    def __init__(self, user_item_matrix, hidden_dim):
        super().__init__(user_item_matrix, hidden_dim)
        self.user_positive_item_dictionary = defaultdict(lambda: [])

        # write list of positive items for each user
        for user, item in zip(self.user_non_zero_idx, self.item_non_zero_idx):
            self.user_positive_item_dictionary[user].append(item)

        self.user_negative_item_dictionary = defaultdict(lambda: [])
        self.unique_user_ids = np.unique(self.user_non_zero_idx)
        unique_item_ids = np.unique(self.item_non_zero_idx)

        # write list of negative items for each user
        for user in self.unique_user_ids:
            self.user_negative_item_dictionary[user] = np.setdiff1d(
                unique_item_ids, self.user_positive_item_dictionary[user])

    def _update_item_vector(self, item_ids: np.array, user_vector: np.array, sigmoid_dot_prod: np.array,
                            gamma: float, lr: float):
        sigmoid_dot_prod = np.einsum('i, ij -> ij', sigmoid_dot_prod, user_vector)
        self.item_matrix[:, item_ids] -= lr * (sigmoid_dot_prod.T + gamma * self.item_matrix[:, item_ids])

    def _update_user_vector(self, user_ids: np.array, error: np.array, lr: float, gamma: float):
        error = np.repeat(error, self.user_matrix.shape[1]).reshape(-1, self.user_matrix.shape[1])
        self.user_matrix[user_ids, :] -= lr * (error + gamma * self.user_matrix[user_ids, :])

    def _update_params(self, user_ids: np.array, positive_item_ids: np.array, negative_item_ids: np.array,
                       gamma: float, lr: float):

        user_items_dot_product = np.einsum(
            'ij, ji -> i', self.user_matrix[user_ids, :],
            self.item_matrix[:, positive_item_ids] - self.item_matrix[:, negative_item_ids])

        sigmoid_user_item_dot = expit(user_items_dot_product)

        self._update_item_vector(positive_item_ids, self.user_matrix[user_ids, :], sigmoid_user_item_dot, gamma, lr)
        self._update_item_vector(negative_item_ids, -self.user_matrix[user_ids, :], sigmoid_user_item_dot, gamma, lr)
        self._update_user_vector(user_ids, sigmoid_user_item_dot * user_items_dot_product, lr, gamma)

    def fit(self, lr: float = 1e-2, batch_size: int = 1, epochs: int = 4, gamma: float = 60):
        n_batches = int(np.ceil(self.unique_user_ids.shape[0] / batch_size))
        start_time = time.time()

        for epoch in range(epochs):
            user_batch_choice = np.random.permutation(self.unique_user_ids)
            error = 0
            for batch in range(n_batches):
                users = user_batch_choice[batch * batch_size:batch * batch_size + batch_size]
                positive_items = np.concatenate(
                    [np.random.choice(self.user_positive_item_dictionary[user], 1) for user in users])
                negative_items = np.concatenate(
                    [np.random.choice(self.user_negative_item_dictionary[user], 1) for user in users])
                self._update_params(users, positive_items, negative_items, gamma, lr)
                if np.dot(
                        self.user_matrix[users, :], self.item_matrix[:, positive_items]) < \
                        np.dot(self.user_matrix[users, :], self.item_matrix[:, negative_items]):
                    error += 1

            if epoch % 1 == 0:
                print(f"Error on epoch {epoch}: {error / n_batches}.")
        print(f"Model is fitted in {int(time.time() - start_time)} seconds.")
        return self.user_matrix, self.item_matrix