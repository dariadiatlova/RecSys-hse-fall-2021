import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
import xgboost as xgb


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
