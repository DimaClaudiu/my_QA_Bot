import ast
import json
import os
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from simpletransformers.classification import ClassificationModel
from simpletransformers.question_answering import QuestionAnsweringModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


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


def build_model():
    model = ClassificationModel('roberta', './outputs/checkpoint-35624-epoch-2', num_labels=10, args={
        'learning_rate': 1e-5, 'num_train_epochs': 4, 'reprocess_input_data': False, 'overwrite_output_dir': True, "train_batch_size": 22,
        "eval_batch_size": 22, "save_steps": 3000, })

    return model


def predict(model, question, mappings):
    inpt = clean_text(question)
    predictions, raw_outputs = model.predict([inpt])

    return mappings[predictions[0]]


def get_best_docs(question, paragraphs, top_k):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 1),
    )

    df = pd.DataFrame.from_dict(paragraphs)

    tfidf_matrix = vectorizer.fit_transform(df["text"])

    question_vec = vectorizer.transform([question])
    scores = tfidf_matrix.dot(question_vec.T).toarray()

    idx_scores = [(idx, score) for idx, score in enumerate(scores)]
    indices_and_scores = OrderedDict(
        sorted(idx_scores, key=(lambda tup: tup[1]), reverse=True)
    )

    df_sliced = df.loc[indices_and_scores.keys()]
    df_sliced = df_sliced[:top_k]

    paragraphs = list(df_sliced.text.values)
    meta_data = [{"date": row["date"]}
                 for idx, row in df_sliced.iterrows()]

    documents = []
    for para, meta in zip(paragraphs, meta_data):
        documents.append(para)

    return documents


def get_labels():
    with open('data/news2/classes.json') as json_file:
        mappings = json.load(json_file)
        inv_map = {v: k for k, v in mappings.items()}

    return inv_map


def main():
    model = build_model()

    # Infer on any question

    print("Ready to roll")
    while True:
        question = input()
        doc = predict(model, question, get_labels())
        print(doc)

        contexts = []
        with open('data/news2/split_contexts/' + doc + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for line in lines:
                context = ast.literal_eval(line)
                contexts.append(context)

            print(len(contexts))

            docs = get_best_docs(question, contexts, 5)


if __name__ == '__main__':
    main()
