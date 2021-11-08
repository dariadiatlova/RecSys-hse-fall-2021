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
        sigmoid_dot_prod = sigmoid_dot_prod * user_vector
        self.item_matrix[:, item_ids] -= lr * (sigmoid_dot_prod.T + gamma * self.item_matrix[:, item_ids])

    def _update_user_vector(self, user_ids: np.array, error: np.array, gamma: float, lr: float):
        self.user_matrix[user_ids, :] -= lr * (error.T + gamma * self.user_matrix[user_ids, :])

    def _update_params(self, user_ids: np.array, positive_item_ids: np.array, negative_item_ids: np.array,
                       gamma: float, lr: float):

        item_embedding_difference = self.item_matrix[:, positive_item_ids] - self.item_matrix[:, negative_item_ids]
        user_items_dot_product = np.dot(self.user_matrix[user_ids, :], item_embedding_difference)
        sigmoid_user_item_dot = expit(user_items_dot_product)

        self._update_item_vector(positive_item_ids, self.user_matrix[user_ids, :], sigmoid_user_item_dot, gamma, lr)
        self._update_item_vector(negative_item_ids, -self.user_matrix[user_ids, :], sigmoid_user_item_dot, gamma, lr)
        self._update_user_vector(user_ids, sigmoid_user_item_dot * item_embedding_difference, gamma, lr)

    def fit(self, lr: float, epochs: int, gamma: float, verbose: int):
        start_time = time.time()

        for epoch in range(epochs):
            user_batch_choice = np.random.permutation(self.unique_user_ids)

            for user in user_batch_choice:
                user_positives = self.user_positive_item_dictionary[user]

                for pos_sample in user_positives:
                    user_negative = np.random.choice(self.user_negative_item_dictionary[user], 1)
                    self._update_params(np.array([user]), np.array([pos_sample]), user_negative, gamma, lr)

        print(f"Model is fitted in {int((time.time() - start_time) / 60)} minutes.")
        return self.user_matrix, self.item_matrix
