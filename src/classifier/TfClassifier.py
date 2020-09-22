from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from transformers import BertConfig, BertTokenizerFast, TFBertModel

from classifier.classifier import Classifier


class PtClassifier(Classifier):

    def __init__(self, model_path=None):
        # loading existing model
        if model_path:
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

        # build tokenizer from bert-base configs
        base_model = 'bert-base-uncased'

        config = BertConfig.from_pretrained(base_model)
        config.output_hidden_states = False

        self.tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=base_model, config=config)

    def predict(self, question, label_mappings):
