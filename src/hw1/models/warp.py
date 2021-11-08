from src.hw1.util import MatrixFactorization
from collections import defaultdict
import numpy as np
import time


class WARP(MatrixFactorization):
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

    def _sample_negative(self, user_id: np.ndarray) -> float:
        user_vector = self.user_matrix[user_id, :]
        self.negative_item = np.random.choice(self.user_negative_item_dictionary[user_id], 1)[0]
        return np.dot(user_vector, self.item_matrix[:, self.negative_item])

    def _update_params(self, user_id: int, positive_item_id: int, negative_item_id: int, lr: float, gamma: float,
                       loss: float):

        user_vector = self.user_matrix[user_id, :]
        positive_item_vector = self.item_matrix[:, positive_item_id]
        negative_item_vector = self.item_matrix[:, negative_item_id]
        positive_negative_item_diff = positive_item_vector - negative_item_vector

        self.user_matrix[user_id, :] -= lr * (loss * positive_negative_item_diff + gamma * user_vector)

        self.item_matrix[:, positive_item_id] -= lr * (loss * user_vector + gamma * positive_item_vector.squeeze())
        self.item_matrix[:, negative_item_id] -= lr * (loss * user_vector + gamma * negative_item_vector.squeeze())

    def fit(self, lr: float = 1e-2, gamma: float = 1e-2, epochs: int = 100, n_negative_samples: int = 10):
        start_time = time.time()
        for epoch in range(epochs):
            users = np.random.permutation(self.unique_user_ids)
            for user in users:
                user_positives = self.user_positive_item_dictionary[user]
                for positive_item in user_positives:
                    positive_dot = np.dot(self.user_matrix[user, :], self.item_matrix[:, positive_item])
                    for i in range(n_negative_samples):
                        negative_dot = self._sample_negative(user)
                        violation = 1.0 + negative_dot - positive_dot
                        if violation <= 0:
                            continue
                        ranking_coefficient = np.floor(n_negative_samples / (i + 1))
                        loss = ranking_coefficient * violation
                        self._update_params(user, positive_item, self.negative_item, lr, gamma, loss)

        print(f"Model is fitted in {int((time.time() - start_time) / 60)} minutes.")
        return self.user_matrix, self.item_matrix
