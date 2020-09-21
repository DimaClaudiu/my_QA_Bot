import ast
import json
from sklearn.model_selection import train_test_split
import pandas as pd

from classifier.classifier import Classifier
from ranker.tfidfRanker import TfidfRanker
from reader.reader import Reader


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

    classifier = Classifier(model_path='models/MCT', num_labels=len(labels))
    ranker = TfidfRanker()
    reader = Reader(model_path='models/QA')

    # Infer on any question
    print("Ready to roll")
    while True:
        question = input()
        group = classifier.predict(question, labels)
        print(group)

        contexts = []
        with open('data/news2/split_contexts/' + group + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for line in lines:
                context = ast.literal_eval(line)
                contexts.append(context)

            print(len(contexts))

            ranked_contexts = ranker.get_best_contexts(question, contexts, 5)

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
