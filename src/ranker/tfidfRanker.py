from collections import OrderedDict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from ranker.ranker import Ranker


class TfidfRanker(Ranker):

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 1),
        )

    def rank(self, question, contexts, top_k):

        df = pd.DataFrame.from_dict(contexts)

        tfidf_matrix = self.vectorizer.fit_transform(df["text"])

        question_vec = self.vectorizer.transform([question])
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
