import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import umap.umap_ as umap
import plotly.express as px


def get_csr_matrix_from_pdf(df: pd.DataFrame, implicit: bool = False, threshold: int = 4):
    """
    Function takes dataframe constructed from MovieLens ratings file, so it has:
    user_id, movie_id and rating columns.
    :param df: pd.DataFrame
    :param implicit: if True, all scores higher or equal than threshold will be converted to ones, others to zeros.
    :param threshold: int value to use for filtering ratings. Default is 4.
    :return: sp.csr.csr_matrix of shape n_unique_users x n_unique_movies.
    """
    if implicit:
        df = df.loc[(df.rating >= threshold)]
        data_vector = np.ones_like(df.user_id)
    else:
        data_vector = df.rating
    user_item = sp.coo_matrix((data_vector, (df.user_id, df.movie_id)))
    return user_item.tocsr()


def visualize_embeddings(embeddings: np.array, movie_info_df: pd.DataFrame):
    u = umap.UMAP().fit(embeddings)
    coords = u.transform(embeddings)
    coords = pd.DataFrame(coords)
    coords = pd.concat([coords.reset_index(drop=True),
                        pd.Series(movie_info_df['category'])], axis=1)
    fig = px.scatter(coords.dropna(), x=0, y=1, color='category')
    return fig


class MatrixFactorization:
    def __init__(self, user_item_matrix: sp.csr_matrix, hidden_dim: int = 16):
        """
        Superclass for initialising user and item matrices for matrix factorization algorithms.
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

    def mse(self) -> float:
        """
        Method returns mse computed on all non_zero values from the target matrix.
        :return: float: mean squared error.
        """
        current_prediction_matrix = np.dot(self.user_matrix, self.item_matrix)
        return np.mean(np.square(current_prediction_matrix[self.user_non_zero_idx, self.item_non_zero_idx]
                                 - self.user_item_matrix.toarray()[self.user_non_zero_idx, self.item_non_zero_idx])
                       )

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

    def recommend(self, user_ids: np.array, n: int = 5):
        recommendations = []
        for i, user in enumerate(user_ids):
            predicted_scores = np.dot(
                self.user_matrix[user, :], self.item_matrix) + self.user_bias[user] + self.item_bias
            recommendations.append(np.argsort(-predicted_scores)[:n])
        return recommendations

    def recall(self, user_ids, prediction):
        matrix = self.user_item_matrix.toarray()
        return np.mean(
            [len(np.intersect1d(prediction[i], np.where(matrix[user] > 0)[0])) / np.sum(
                matrix[user]) for i, user in enumerate(user_ids)]
        )
