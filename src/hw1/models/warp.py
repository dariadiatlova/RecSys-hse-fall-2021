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
        self.negative_item = np.random.choice(self.user_negative_item_dictionary[user_id], 1)
        return np.dot(user_vector, self.item_matrix[:, self.negative_item])

    def _update_params(self, user_id: int, positive_item_id: int, negative_item_id: int, lr: float, gamma: float,
                       ranking_coefficient: float):

        user_vector = self.user_matrix[user_id, :]
        positive_item_vector = self.item_matrix[:, positive_item_id]
        negative_item_vector = self.item_matrix[:, negative_item_id]

        self.user_matrix[user_id, :] -= lr * (ranking_coefficient * (negative_item_vector - positive_item_vector).T +
                                              gamma * user_vector)[0]

        self.item_matrix[:, positive_item_id] -= lr * (
                -ranking_coefficient * user_vector + gamma * positive_item_vector.squeeze()).reshape(-1, 1)
        self.item_matrix[:, negative_item_id] -= lr * (
                ranking_coefficient * user_vector + gamma * negative_item_vector.squeeze()).reshape(-1, 1)

    def fit(self, lr: float = 1e-2, gamma: float = 1e-2, epochs: int = 100, verbose: int = 100):
        start_time = time.time()
        for epoch in range(epochs):
            n_attempts = []
            users = np.random.permutation(self.unique_user_ids)

            for user in users:
                positive_item = np.random.choice(self.user_positive_item_dictionary[user], 1)
                positive_dot = np.dot(self.user_matrix[user, :], self.item_matrix[:, positive_item])
                negative_dot = self._sample_negative(user)
                n_attempts_to_sample_wrong = 1
                user_negative_samples_count = len(self.user_negative_item_dictionary[user])
                while positive_dot > negative_dot and n_attempts_to_sample_wrong < user_negative_samples_count:
                    n_attempts_to_sample_wrong += 1
                    negative_dot = self._sample_negative(user)
                ranking_coefficient = np.floor((user_negative_samples_count - 1) / n_attempts_to_sample_wrong)
                self._update_params(user, positive_item, self.negative_item, lr, gamma, ranking_coefficient)

                if epoch % verbose == 0:
                    n_attempts.append(n_attempts_to_sample_wrong)

            if epoch % verbose == 0:
                print(f"Mean number of attempts to sample wrong on epoch {epoch}: {np.mean(n_attempts)}.")

        print(f"Model is fitted in {int((time.time() - start_time) / 60)} minutes.")
        return self.user_matrix, self.item_matrix
