from reader.TfReader import TfReader


import json
from pathlib import Path


def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts[0:1000], questions[0:1000], answers[0:1000]


train_contexts, train_questions, train_answers = read_squad(
    'data/squads/train-v2.0.json')
val_contexts, val_questions, val_answers = read_squad(
    'data/squads/dev-v2.0.json')


model = TfReader()

model.train(train_contexts, train_questions, train_answers,
            epochs=1, save_path='./mydistilBERT2', batch_size=5)

model.load('./mydistilBERT2')
model.eval(val_contexts, val_questions,
           val_answers, batch_size=5)
