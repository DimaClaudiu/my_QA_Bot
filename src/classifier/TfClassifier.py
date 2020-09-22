from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from transformers import BertConfig, BertTokenizerFast, TFBertModel

from classifier.classifier import Classifier

import numpy as np


class TfClassifier(Classifier):

    def __init__(self, model_path=None, max_token_length=64):
        # loading existing model
        self.max_length = max_token_length
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
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=self.metric)

        # build tokenizer from bert-base configs
        base_model = 'bert-base-uncased'

        config = BertConfig.from_pretrained(base_model)
        config.output_hidden_states = False

        self.tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=base_model, config=config)

    def predict(self, question, label_mappings):
        tokens = self.tokenizer(
            text=[question],
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors='tf',
            return_token_type_ids=False,
            return_attention_mask=True,
            verbose=False)

        result = self.model.predict(
            x={'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']})

        ranked_classes = np.argsort(result['classes'][0])

        return ranked_classes[len(ranked_classes) - 1]
