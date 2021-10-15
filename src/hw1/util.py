import pandas as pd
import scipy.sparse as sp


def get_csr_matrix_from_pdf(df: pd.DataFrame):
    """
    Function takes dataframe constructed from MovieLens ratings file, so it has:
    user_id, movie_id and rating columns.
    :param df: pd.DataFrame
    :return: sp.csr.csr_matrix of shape n_unique_users x n_unique_movies.
    """
    user_item = sp.coo_matrix((df.rating, (df.user_id, df.movie_id)))
    user_item_t_csr = user_item.T.tocsr()
    return user_item.tocsr()
