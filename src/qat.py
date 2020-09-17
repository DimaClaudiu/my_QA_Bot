from simpletransformers.question_answering import QuestionAnsweringModel
from sklearn.model_selection import train_test_split
import json
import os
import logging
import ast


def read_data(filename, test_size_split=0.15):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []

    for line in lines:
        data.append(ast.literal_eval(line))

    train_data, eval_data = train_test_split(data, test_size=test_size_split)

    return train_data, eval_data


def main():
    train_data, eval_data = read_data('data/squad2/clean_train.txt')

    model = QuestionAnsweringModel('roberta', 'roberta-base', args={
                                   'reprocess_input_data': True, 'overwrite_output_dir': True, "train_batch_size": 6})

    model.train_model(train_data)

    result, text = model.eval_model(train_data)

    print(result)


if __name__ == '__main__':
    main()
