import numpy as np
import pandas as pd
import xgboost as xgb

from omegaconf import OmegaConf
from tqdm import tqdm
from typing import List

from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

from src.hw2.configs import CONFIG_PATH
from src.hw2.embedding_dataset import Dataset
from src.hw2.preprocessing import get_train_df
from src.hw2.utils import to_categorical


class SimpleXgbRecommender:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __set_k_fold(self):
        return KFold(n_splits=self.k)

    @staticmethod
    def evaluate(model, test_data, test_target):
        prediction_probabilities = model.predict_proba(test_data)
        fpr, tpr, thresholds = metrics.roc_curve(test_target, prediction_probabilities[:, 1], pos_label=1)
        return metrics.auc(fpr, tpr)

    def train(self, data, target):
        main_train_array = np.array(data)
        main_train_target = np.array(target)
        k_fold_auc = []

        for train_index, test_index in self.__set_k_fold().split(data, target):
            train_data = main_train_array[train_index]
            train_target = main_train_target[train_index]
            test_data = main_train_array[test_index]
            test_target = main_train_target[test_index]

            model = xgb.XGBClassifier(learning_rate=self.learning_rate,
                                      max_depth=self.max_depth,
                                      min_child_weight=self.min_child_weight,
                                      n_estimators=self.n_estimators,
                                      eval_metric=self.eval_metric,
                                      objective="binary:logistic",
                                      use_label_encoder=False)

            model.fit(train_data, train_target)
            k_fold_auc.append(self.evaluate(model, test_data, test_target))
        return k_fold_auc, model.feature_importances_


class KNNRecommender:
    def __init__(self, n_neighbors: int = 150, algorithm: str = "auto"):
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm)

    def fit(self, all_songs_embeddings: np.ndarray):
        self.nbrs.fit(all_songs_embeddings)

    def predict(self, embeddings: np.ndarray, return_distances: bool = False):
        return self.nbrs.kneighbors(embeddings, return_distance=return_distances)


class KNNXGB:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def train(self, n_train_samples: int = None, evaluate: bool = False):
        print(f"Filtering data...")
        train_df = get_train_df()
        if n_train_samples:
            train_df = train_df.sample(n=n_train_samples)
        dataset = Dataset(train_df)
        song_ids = dataset.get_song_ids()

        print(f"Tokenize data...")
        df_all_songs_emb = dataset.tokenize(emb_size=self.emb_size)
        song_embedding_ser = df_all_songs_emb["song_embedding"].values
        cat_features_df = to_categorical(df_all_songs_emb.drop(columns=["song_embedding"]))
        cat_features_df["song_embedding"] = song_embedding_ser
        df_user_emb = dataset.group_by_song(cat_features_df, only_positive=True)
        # drop users whom we lost while computing embeddings
        cat_features_df = pd.merge(df_user_emb, cat_features_df, how="inner").drop(columns=['avg_song_emb'])
        all_songs_embeddings = np.array(list(song_embedding_ser))

        print(f"Started KNN training...")
        self.knn = KNNRecommender(self.n_neighbors, self.algorithm)
        print(f"Songs embedding count: {len(all_songs_embeddings)}.")
        self.knn.fit(all_songs_embeddings)

        cat_features_df_with_target = cat_features_df.copy()
        target = cat_features_df.pop("target")
        data_train, data_test, target_train, target_test = train_test_split(cat_features_df, target.values,
                                                                            test_size=self.test_size)
        print(f"Started XGB training...")
        self.xgb = xgb.XGBClassifier(learning_rate=self.learning_rate,
                                      max_depth=self.max_depth,
                                      min_child_weight=self.min_child_weight,
                                      n_estimators=self.n_estimators,
                                      eval_metric=self.eval_metric,
                                      objective="binary:logistic",
                                      use_label_encoder=False)

        xgb_train_data = np.array(data_train.drop(columns=["song_embedding"]).values)
        self.xgb.fit(xgb_train_data, np.array(target_train))

        # for prediction & evaluation
        self.cat_features_df_predict = cat_features_df_with_target
        self.song_ids = song_ids
        self.user_emb = df_user_emb
        self.data_with_artist_name = train_df

        if evaluate:
            return self.evaluation(data_test)

    def evaluation(self, data_test):
        print("Getting KNN predictions...")

        user_embs = self.user_emb.set_index("msno").loc[data_test.msno.values].avg_song_emb
        candidates = self.knn.predict(np.array(list(user_embs)))

        metric_measure = []
        all_targets = []
        all_predictions = []
        cat_features_view = self.cat_features_df_predict.reset_index(drop=True)

        for i, user in enumerate(tqdm(data_test.msno.values, desc="Evaluated:")):
            user_predictions = []
            user_songs = np.array(
                self.cat_features_df_predict[self.cat_features_df_predict.msno == user].song_id.values.tolist())

            for candidate in candidates[i]:
                song_id = self.song_ids[candidate]

                if song_id in user_songs:
                    df = cat_features_view.loc[(cat_features_view["msno"] == user) & (cat_features_view["song_id"] == song_id)]
                    target = int(list(df.pop("target"))[0])
                    score = self.xgb.predict_proba(df.drop(columns=["song_embedding"]).values)[:, 1]
                    user_predictions.append([target, score[0]])

            if user_predictions:
                predictions = np.sort(np.array(user_predictions), axis=0)[::-1]
                recall = np.sum(predictions[:self.n_predictions, 0]) / len(predictions[:self.n_predictions, 0])
                all_targets.append(predictions[:self.n_predictions, 0])
                all_predictions.append(predictions[: self.n_predictions, 1])
                metric_measure.append(recall)

        fpr, tpr, thresholds = metrics.roc_curve(np.concatenate(all_targets).flatten(),
                                                 np.concatenate(all_predictions).flatten(), pos_label=1)
        auc = metrics.auc(fpr, tpr)

        return np.mean(metric_measure), auc

    def predict(self, user_ids: List[int]):
        artist_names = np.array(list(self.data_with_artist_name["artist_name"].values))
        user_embs = self.user_emb.set_index("msno").loc[pd.Series(user_ids)].avg_song_emb
        candidates = self.knn.predict(np.array(list(user_embs)))
        cat_features_view = self.cat_features_df_predict.reset_index(drop=True)

        to_recommend = int(self.n_recommendations * (1 - self.exploration_rate))
        to_sample = int(self.n_recommendations - to_recommend)

        user_recommendation_list = []
        user_true_songs = []

        for i, user_id in enumerate(tqdm(user_ids, desc="Predicted:")):
            user_prediction_scores = []
            user_true_artists = []

            user_songs = np.array(
                self.cat_features_df_predict.loc[(self.cat_features_df_predict.msno == user_id) &
                                                 (self.cat_features_df_predict.target == 1)].song_id.values.tolist())

            for song_id in candidates[i]:
                if song_id in user_songs:
                    df = cat_features_view.loc[(cat_features_view["msno"] == user_id) &
                                               (cat_features_view["song_id"] == song_id)]
                    score = self.xgb.predict_proba(df.drop(columns=["song_embedding", "target"]).values)[:, 1]
                    user_prediction_scores.append([score, artist_names[song_id]])

            if user_prediction_scores:
                if len(user_prediction_scores) < to_recommend:
                    to_sample = self.n_recommendations - len(user_prediction_scores)
                sorted_predictions = np.sort(np.array(user_prediction_scores), axis=0)[::-1]
                user_recommendations = np.concatenate(sorted_predictions[:to_recommend, 1],
                                                      artist_names[np.random.choice(candidates[i], to_sample)])
            else:
                user_recommendations = np.random.choice(artist_names[candidates[i]], self.n_recommendations)

            if len(user_songs) > self.n_recommendations:
                user_true_artists.append(artist_names[np.random.choice(user_songs, self.n_recommendations)])
            else:
                user_true_artists.append(artist_names[user_songs])

            user_recommendation_list.append(user_recommendations)
            user_true_songs.append(user_true_artists[0])

        return user_recommendation_list, user_true_songs


# config = OmegaConf.load(CONFIG_PATH)
# config = OmegaConf.to_container(config, resolve=True)
# alg = KNNXGB(**config)
# alg.train(n_train_samples=100_000, evaluate=False)
# print(alg.predict([1, 456, 689]))
