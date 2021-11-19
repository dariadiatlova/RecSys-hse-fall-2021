import pandas as pd
import matplotlib.pyplot as plt

from os.path import join
from typing import List
from src.hw2 import DATA_ROOT


def read_dfs(file_names: List[str]) -> List[pd.DataFrame]:
    return [pd.read_csv(join(DATA_ROOT, name)) for name in file_names]


def to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')

    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].cat.codes
    return df


def group_by_plot(df, feature_to_group, feature_to_count, title, xname, yname):
    city_count = df.groupby(feature_to_group).count()[feature_to_count]
    city_id = df.groupby(feature_to_group).count().index.tolist()
    plt.figure()
    plt.plot(city_id, city_count)
    plt.title(title)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()
