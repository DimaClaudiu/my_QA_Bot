import ast
import json
from sklearn.model_selection import train_test_split
import pandas as pd

from classifier.StClassifier import StClassifier
from classifier.TfClassifier import TfClassifier
from ranker.tfidfRanker import TfidfRanker
from reader.StReader import StReader
from reader.PtReader import PtReader
from reader.TfReader import TfReader


def read_data(path, test_size_split=0.15):
    df = pd.read_csv(path)

    train_df, eval_df = train_test_split(df, test_size=test_size_split)

    return train_df, eval_df


def get_labels():
    with open('data/news2/classes.json') as json_file:
        mappings = json.load(json_file)
        inv_map = {v: k for k, v in mappings.items()}

    return inv_map


def main():

    labels = get_labels()

    classifier = TfClassifier()
    classifier.load('models/MCT/tfmct1/tfmlc.h5')
    ranker = TfidfRanker()
    reader = TfReader(
        model_path='models/QA/tf_bert_squad1')

    # Infer on any question
    print("Ready to roll")
    while True:
        question = input()
        label_id = classifier.predict(question)
        group = labels[label_id]
        print(group)

        contexts = []
        with open('data/news2/split_contexts/' + group + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for line in lines:
                context = ast.literal_eval(line)
                contexts.append(context)

            print(len(contexts))

            ranked_contexts = ranker.rank(question, contexts, 5)

            for context in ranked_contexts:
                print(context['text'])
                answer, probability = reader.predict(
                    question, context['text'])

                if answer != '':
                    print(f'^^ {answer} - {probability} ^^\n')


if __name__ == '__main__':
    main()
