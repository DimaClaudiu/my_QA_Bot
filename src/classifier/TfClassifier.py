from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.initializers import TruncatedNormal

from transformers import BertConfig, BertTokenizerFast, TFBertModel

from classifier.classifier import Classifier

import numpy as np


class TfClassifier(Classifier):

    def __init__(self, model_path=None, max_token_length=64):
        self.max_length = max_token_length

        self.optimizer = Adam(
            learning_rate=5e-05,
            epsilon=1e-08,
            decay=0.01,
            clipnorm=1.0)

        self.loss = {'classes': CategoricalCrossentropy(
            from_logits=True)}

        self.metric = {'classes': CategoricalAccuracy(
            'accuracy')}

        # build tokenizer from bert-base configs
        self.base_model = 'bert-base-uncased'

        self.config = BertConfig.from_pretrained(base_model)
        self.config.output_hidden_states = False

        self.tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=base_model, config=config)

    def create(self, num_classes=10, model_name='BERT_MCC'):
        transformer_model = TFBertModel.from_pretrained(
            self.base_model, config=self.config)
        bert = transformer_model.layers[0]

        input_ids = Input(shape=(self.max_length,),
                          name='input_ids', dtype='int32')

        attention_mask = Input(shape=(self.max_length,),
                            name='attention_mask', dtype='int32')
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

        bert_model = bert(inputs)[1]

        dropout = Dropout(self.config.hidden_dropout_prob,
                          name='pooled_output')
        pooled_output = dropout(bert_model, training=False)

        classes = Dense(units=num_classes, kernel_initializer=TruncatedNormal(
    stddev=self.config.initializer_range), name='classes')(pooled_output)

        outputs = {'classes': classes}

        model = Model(inputs=inputs, outputs=outputs,
              name=model_name)

        model.summary()


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

    @staticmethod
    def load(model_path):
        self.model = load_model(model_path)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metric)
