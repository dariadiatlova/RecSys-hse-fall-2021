from src.hw2.utils import read_dfs, to_categorical
import pandas as pd


def get_train_df(convert_to_categorical: bool = False):
    file_names = ["members.csv", "sample_submission.csv", "song_extra_info.csv", "songs.csv", "test.csv", "train.csv"]
    members_df, submissions_df, songs_extra_info_df, songs_df, test_df, train_df = read_dfs(file_names)

    # members df cleaning
    members_df_clean = members_df.drop(columns=["gender"])
    bd_mask = members_df_clean.bd != 0
    members_df_clean.loc[bd_mask, "bd"] = 1
    city_mask = members_df_clean.city != 1
    members_df_clean.loc[city_mask, "city"] = 0

    # songs df cleaning
    songs_df_clean = songs_df.drop(columns=["lyricist"])
    songs_df_clean = songs_df_clean.fillna("UNK")

    # merged df cleaning
    train_df = train_df.merge(members_df_clean, on="msno")
    train_df = train_df.merge(songs_df_clean, on="song_id")
    train_df = train_df.fillna(value="unknown")

    train_df.registration_init_time = pd.to_datetime(
        train_df.registration_init_time, format='%Y%m%d', errors='ignore').dt.year
    train_df.expiration_date = pd.to_datetime(train_df.expiration_date, format='%Y%m%d', errors='ignore').dt.year
    train_df = train_df.drop(columns=["bd", "source_screen_name", "registration_init_time"])

    if convert_to_categorical:
        train_df = to_categorical(train_df)
    return train_df
