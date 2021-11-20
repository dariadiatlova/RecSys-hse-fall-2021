import numpy as np
import xgboost as xgb
from omegaconf import OmegaConf

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

    def train(self):
        print(f"Filtering data...")
        train_df = get_train_df()
        train_df = train_df.sample(n=500)
        dataset = Dataset(train_df)

        print(f"Tokenize data...")
        df_all_songs_emb = dataset.tokenize(emb_size=self.emb_size)
        song_embedding_ser = df_all_songs_emb["song_embedding"].values
        cat_features_df = to_categorical(df_all_songs_emb.drop(columns=["song_embedding"]))
        cat_features_df["song_embedding"] = song_embedding_ser
        df_user_emb = dataset.group_by_song(cat_features_df)
        all_songs_embeddings = np.array(list(song_embedding_ser))

        print(f"Started KNN training...")
        self.knn = KNNRecommender(self.n_neighbors, self.algorithm)
        self.knn.fit(all_songs_embeddings)

        target = cat_features_df.pop("target")
        data_train, data_test, target_train, target_test = train_test_split(cat_features_df, target,
                                                                            test_size=self.test_size)
        print(f"Started XGB training...")
        self.xgb = xgb.XGBClassifier(learning_rate=self.learning_rate,
                                      max_depth=self.max_depth,
                                      min_child_weight=self.min_child_weight,
                                      n_estimators=self.n_estimators,
                                      eval_metric=self.eval_metric,
                                      objective="binary:logistic",
                                      use_label_encoder=False)

        xgb_train_data = np.array(data_train.drop(columns=["song_embedding"]))
        self.xgb.fit(xgb_train_data, np.array(target_train))
        return self.evaluation(data_test, target_test, df_user_emb, cat_features_df, train_df)

    def evaluation(self, data_test, target_test, df_user_emb, cat_features_df, train_df):
        user_embs = df_user_emb.set_index("msno").loc[data_test.msno.values].avg_song_emb
        candidates = self.knn.predict(np.array(list(user_embs)))

        all_user_predictions = []
        all_users = []
        cat_features_view = cat_features_df.reset_index(drop=True)
        train_df_view = train_df.reset_index(drop=True)

        for i, user in enumerate(data_test.msno.values):
            user_predictions = []
            user_songs = np.array(cat_features_df[cat_features_df.msno == user].song_id.values.tolist())

            for candidate in candidates[i]:
                song_id = cat_features_view.loc[candidate].song_id

                if song_id in user_songs:
                    df = cat_features_view.loc[(cat_features_view["msno"] == user) & (cat_features_view["song_id"] == song_id)]
                    score = self.xgb.predict_proba(df.drop(columns=["song_embedding"]))[:, 1]
                    user_predictions.append([song_id, score[0]])

            if user_predictions:
                predictions = np.sort(np.array(user_predictions), axis=0)[::-1]
                all_users.append(user)
                all_user_predictions.append(predictions[:self.n_predictions, 0])

        return all_users, all_user_predictions


config = OmegaConf.load(CONFIG_PATH)
config = OmegaConf.to_container(config, resolve=True)
alg = KNNXGB(**config)
cnd = alg.train()
print(cnd)
