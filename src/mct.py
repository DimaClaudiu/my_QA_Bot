from simpletransformers.classification import ClassificationModel
import pandas as pd
from sklearn.model_selection import train_test_split

import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import random


def clean_text(text, max_len=128):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    words = [word for word in word_tokens if word.isalpha()]

    filtered_words = [w for w in words if not w in stop_words]

    return ' '.join(filtered_words)[0:max_len-1].lower()


def read_data(path, test_size_split=0.15):
    df = pd.read_csv(path)

    train_df, eval_df = train_test_split(df, test_size=test_size_split)

    return train_df, eval_df


def main():
    train_df, eval_df = read_data('./data/news2/dataset/dataset_proccesed.csv')


if __name__ == '__main__':
    main()
