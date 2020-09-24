import ast
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from classifier.TfClassifier import TfClassifier
from ranker.tfidfRanker import TfidfRanker
from reader.PtReader import PtReader
from reader.TfReader import TfReader


from silence_tensorflow import silence_tensorflow
silence_tensorflow()


LABEL_PATH = 'chatBot/data/news2/classes.json'
DB_PATH = 'chatBot/data/news2/split_contexts/'

CLASSIFIER_PATH = 'chatBot/models/MCT/tf_bert_base_1.h5'
READER_PATH = 'chatBot/models/QA/electra_squad2'


def read_data(path, test_size_split=0.15):
    df = pd.read_csv(path)

    train_df, eval_df = train_test_split(df, test_size=test_size_split)

    return train_df, eval_df


def get_labels():
    with open(LABEL_PATH) as json_file:
        mappings = json.load(json_file)
        inv_map = {v: k for k, v in mappings.items()}

    return inv_map


input_lock = False


def process_input(shown):
    global input_lock
    if input_lock or shown == 0 or shown % 5 != 0:
        return 0

    print('s: show more, n: new question, q: quit')

    inpt = input()

    if inpt.lower() == 's':
        input_lock = True
        return 0
    if inpt.lower() == 'n':
        return 1
    elif inpt.lower() == 'q':
        exit(code=0)
    else:
        process_input(shown)
        return 0


def main():

    labels = get_labels()

    classifier = TfClassifier()
    classifier.load(CLASSIFIER_PATH)
    ranker = TfidfRanker()
    reader = TfReader()
    reader.load(
        model_path=READER_PATH)

    # Infer on any question
    while True:
        print("Ready to roll")
        question = input()
        label_id = classifier.predict(question)
        group = labels[label_id]
        print(group)

        contexts = []

        shown = 0
        predicted = 0
        global input_lock
        with open(DB_PATH + group + '.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for line in lines:
                context = ast.literal_eval(line)
                contexts.append(context)

            print(len(contexts))

            ranked_contexts = ranker.rank(question, contexts, 0)

            for _, row in ranked_contexts.iterrows():
                date = row['date']
                text = row['text']

                answer, probability = reader.predict(
                    question, text)
                predicted += 1

                if answer != '':
                    pretty_probability = str(probability*100)[0:3] + '%'

                    print(f'{date}: {answer} - {pretty_probability}')
                    print(f'context: "{text}"\n')

                    input_lock = False
                    shown += 1

                elif predicted % 50 == 0:
                    print(predicted)

                new_question = process_input(shown)
                if new_question:
                    break


if __name__ == '__main__':
    main()
