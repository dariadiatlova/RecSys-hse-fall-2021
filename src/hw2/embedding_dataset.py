import pandas as pd
import numpy as np
from src.hw2.tokenizer.tokenize import load_tokenizer


class Dataset:
    def __init__(self, df):
        self.df = df

    def tokenize(self, emb_size: int = 10) -> pd.DataFrame:
        tokenizer = load_tokenizer()
        tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=emb_size)
        tokenizer.enable_truncation(emb_size)
        self.df["artist_name_embedding"] = pd.Series(
            [tokenizer.encode(artist_name).ids for artist_name in self.df.artist_name]).values
        self.df["composer_embedding"] = pd.Series(
            [tokenizer.encode(composer).ids for composer in self.df.composer]).values
        self.df["song_embedding"] = self.df["artist_name_embedding"].values + self.df["composer_embedding"].values
        return self.df.drop(columns=["composer_embedding", "artist_name_embedding"])

    @staticmethod
    def group_by_song(df: pd.DataFrame) -> pd.DataFrame:
        songs = df.groupby("msno")["song_embedding"].apply(lambda x: np.array(x.tolist()).mean(axis=0))
        return pd.DataFrame({'msno': songs.index, 'avg_song_emb': songs.values})
