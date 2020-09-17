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


def build_model():
    model = ClassificationModel('roberta', './outputs/checkpoint-17812-epoch-1', num_labels=10, args={
        'learning_rate': 1e-5, 'num_train_epochs': 4, 'reprocess_input_data': True, 'overwrite_output_dir': True, "train_batch_size": 22,
        "eval_batch_size": 22, "save_steps": 1000, })

    return model


def predict(model, question, mappings):
    inpt = clean_text(question)
    predictions, raw_outputs = model.predict([inpt])

    return mappings[predictions[0]]


def main():
    # Read data and build model
    train_df, eval_df = read_data('./data/news2/dataset/dataset_proccesed.csv')
    model = build_model()

    # Train model
    model.train_model(train_df)

    # Evaluate after training
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    print(result)

    # Infer on any question
    with open('data/news2/classes.json') as json_file:
        mappings = json.load(json_file)
        inv_map = {v: k for k, v in mappings.items()}

    print("Ready to roll")
    while True:
        print(predict(model, input(), inv_map))


if __name__ == '__main__':
    main()
