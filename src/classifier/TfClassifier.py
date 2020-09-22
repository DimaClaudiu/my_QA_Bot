from classifier.classifier import Classifier
from transformers import TFBertModel,  BertConfig, BertTokenizerFast
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy


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

    def predict(self, question, label_mappings):
