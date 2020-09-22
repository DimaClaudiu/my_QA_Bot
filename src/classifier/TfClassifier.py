from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from transformers import BertConfig, BertTokenizerFast, TFBertModel

from classifier.classifier import Classifier


class PtClassifier(Classifier):

    def __init__(self, model_path):
        self.optimizer = Adam(
            learning_rate=5e-05,
            epsilon=1e-08,
            decay=0.01,
            clipnorm=1.0)

        self.loss = {'classes': CategoricalCrossentropy(
            from_logits=True)}

        self.metric = {'classes': CategoricalAccuracy(
            'accuracy')}

        self.model = load_model(model_path)

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metric)

    def predict(self, question, label_mappings):
