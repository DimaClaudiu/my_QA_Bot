import ast
import json
import os
import random
from collections import OrderedDict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from reader.reader import Reader
from classifier.classifier import Classifier


def read_data(path, test_size_split=0.15):
    df = pd.read_csv(path)

    train_df, eval_df = train_test_split(df, test_size=test_size_split)

    return train_df, eval_df


def get_best_contexts(question, contexts, top_k):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 1),
    )

    df = pd.DataFrame.from_dict(contexts)

    tfidf_matrix = vectorizer.fit_transform(df["text"])

    question_vec = vectorizer.transform([question])
    scores = tfidf_matrix.dot(question_vec.T).toarray()

    idx_scores = [(idx, score) for idx, score in enumerate(scores)]
    indices_and_scores = OrderedDict(
        sorted(idx_scores, key=(lambda tup: tup[1]), reverse=True)
    )

    df_sliced = df.loc[indices_and_scores.keys()]
    df_sliced = df_sliced[:top_k]

    conte = list(df_sliced.text.values)
    meta_data = [{"date": row["date"]} for _, row in df_sliced.iterrows()]

    ranked_contexts = []
    for para, meta in zip(contexts, meta_data):
        ranked_contexts.append(para)

    return ranked_contexts


def get_labels():
    with open('data/news2/classes.json') as json_file:
        mappings = json.load(json_file)
        inv_map = {v: k for k, v in mappings.items()}

    return inv_map


def main():

    labels = get_labels()
    ranker = Classifier(model_path='models/MCT', num_labels=len(labels))
    reader = Reader(model_path='models/QA')

    # Infer on any question
    print("Ready to roll")
    while True:
        question = input()
        group = ranker.predict(question, labels)
        print(group)

        contexts = []
        with open('data/news2/split_contexts/' + group + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for line in lines:
                context = ast.literal_eval(line)
                contexts.append(context)

            print(len(contexts))

            ranked_contexts = get_best_contexts(question, contexts, 5)
            for context in ranked_contexts:
                answers, probabilities = reader.predict(
                    question, context['text'])

                for i in range(len(answers)):
                    pair = answers[i]
                    for j in range(len(pair['answer'])):
                        print(pair['id'] + ": " + pair['answer']
                              [j] + " - " + str(probabilities[i]['probability'][j]))
                print('\n')


if __name__ == '__main__':
    main()
