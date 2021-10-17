import numpy as np
import time
from src.hw1.util import MatrixFactorization
from collections import defaultdict
from typing import List


class ALS(MatrixFactorization):
    def __init__(self, user_item_matrix, hidden_dim):
        super().__init__(user_item_matrix, hidden_dim)
        self.user_item_dictionary = defaultdict(lambda: [])
        self.item_user_dictionary = defaultdict(lambda: [])
        for user, item in zip(self.user_non_zero_idx, self.item_non_zero_idx):
            self.user_item_dictionary[user].append(item)
            self.item_user_dictionary[item].append(user)
        self.identity_matrix = np.identity(hidden_dim)

    def _inverse_user_matrix(self, items: List[int], gamma: float):
        item_product_sum = np.sum(np.einsum('ij, ji -> i', self.item_matrix[:, items].T, self.item_matrix[:, items]))
        inverse_item_matrix = np.linalg.inv(item_product_sum + gamma * self.identity_matrix)
        return inverse_item_matrix

    def _inverse_item_matrix(self, users: List[int], gamma: float):
        user_product_sum = np.sum(np.einsum('ij, ji -> i', self.user_matrix[users, :], self.user_matrix[users, :].T))
        inverse_user_matrix = np.linalg.inv(user_product_sum + gamma * self.identity_matrix)
        return inverse_user_matrix

    def fit(self, gamma: float = 200, epochs: int = 3):

        mse_logging = []
        start_time = time.time()

        for epoch in range(epochs):

            users = np.random.permutation(list(self.user_item_dictionary.keys()))
            # update user vector
            for user in users:
                items = self.user_item_dictionary[user]
                inverse_user_matrix = self._inverse_user_matrix(items, gamma)

                predictions = np.einsum('i, ji -> j', self.user_matrix[user, :], self.item_matrix[:, items].T)

                self.user_matrix[user, :] = np.dot(
                    inverse_user_matrix, np.sum(
                        np.einsum('i, ji -> ji', predictions, self.item_matrix[:, items]), axis=1
                    ))

            items = np.random.permutation(list(self.item_user_dictionary.keys()))
            # update item vector
            for item in items:
                users = self.item_user_dictionary[item]
                inverse_item_matrix = self._inverse_item_matrix(users, gamma)

                predictions = np.einsum('i, ji -> j', self.item_matrix[:, item], self.user_matrix[users, :])

                self.item_matrix[:, item] = np.dot(
                    inverse_item_matrix, np.sum(
                        np.einsum('i, ij -> ij', predictions, self.user_matrix[users, :]), axis=0
                    ))
            mse = self.mse()
            mse_logging.append(mse)

            print(f"Mse on epoch {epoch}: {mse}.")

        print(f"Model fitted in: {int(time.time() - start_time)} seconds.")

        return self.user_matrix, self.item_matrix
