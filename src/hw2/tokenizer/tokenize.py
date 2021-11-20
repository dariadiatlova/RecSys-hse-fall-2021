import pandas as pd
import numpy as np

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from src.hw2.tokenizer.__init__ import TOKENIZER_PATH


def train_tokenizer(data: np.ndarray, data_file_name: str = "data.raw",
                    tokenizer_file_name: str = "songs_tokenizer.json"):

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()

    pd.DataFrame({'token': data}).to_csv(TOKENIZER_PATH / data_file_name, index=False)
    tokenizer.train([str(TOKENIZER_PATH / data_file_name)], trainer)
    tokenizer.save(str(TOKENIZER_PATH / tokenizer_file_name))


def load_tokenizer(trained_tokenizer_path: str = str(TOKENIZER_PATH / "songs_tokenizer.json")):
    return Tokenizer.from_file(trained_tokenizer_path)
