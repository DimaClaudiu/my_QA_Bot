from classifier.TfClassifier import TfClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def demo_train():
    data = pd.read_csv('data/news2/dataset/dataset_proccesed.csv')
    data = data[0:1000]

    data = data.dropna()

    data, data_test = train_test_split(
        data, test_size=0.2)

    model = TfClassifier()
    model.create()

    x_train = data['text'].to_list()
    y_train = to_categorical(data['labels'])

    model.train(x_train, y_train, './demoModel.h5', epochs=1)


def demo_predict():
    model = TfClassifier()
    model.load('demoModel.h5')

    result = model.predict('Who is the president?')

    print(result)


if __name__ == '__main__':
    demo_train()

    demo_predict()
