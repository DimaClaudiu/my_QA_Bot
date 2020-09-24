import math

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
from transformers import TFDistilBertForQuestionAnswering
from transformers import DistilBertTokenizerFast
from reader.reader import Reader


class TfReader(Reader):

    def load(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = TFAutoModelForQuestionAnswering.from_pretrained(
            model_path)
        self.model.summary()

    def predict(self, question, context, best_n=3):

        inputs = self.tokenizer(
            question, context, add_special_tokens=True, return_tensors="tf")

        input_ids = inputs["input_ids"].numpy()[0]

        text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        answer_start_scores, answer_end_scores = self.model(inputs)

        answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
        answer_end = (tf.argmax(answer_end_scores, axis=1) + 1).numpy()[0]

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        probabilities = []

        for start_score in answer_start_scores.cpu().numpy():
            probabilities.append(self._compute_softmax(start_score))

        probability = (probabilities[0][answer_start] +
                       probabilities[0][answer_end - 1]) / 2

        return answer, probability

    def train(self, train_contexts, train_questions, train_answers, base_model='distilbert-base-uncased', epochs=10, batch_size=10, save_path='.'):

        tokenizer = DistilBertTokenizerFast.from_pretrained(
            base_model)

        self._add_end_idx(train_answers, train_contexts)

        train_encodings = tokenizer(
            train_contexts, train_questions, truncation=True, padding=True)

        self._add_token_positions(train_encodings, train_answers, tokenizer)

        train_dataset = tf.data.Dataset.from_tensor_slices((
            {key: train_encodings[key]
                for key in ['input_ids', 'attention_mask']},
            {key: train_encodings[key]
                for key in ['start_positions', 'end_positions']}
        ))

        self.model = TFDistilBertForQuestionAnswering.from_pretrained(
            base_model)

        # Keras will expect a tuple when dealing with labels
        train_dataset = train_dataset.map(lambda x, y: (
            x, (y['start_positions'], y['end_positions'])))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.distilbert.return_dict = False

        # fit model
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.fit(train_dataset.shuffle(
            1000).batch(batch_size), epochs=epochs, batch_size=batch_size)

        self.model.save_pretrained(save_path)

    def eval(self, val_contexts, val_questions, val_answers, base_model='distilbert-base-uncased', batch_size=10):
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            base_model)

        self._add_end_idx(val_answers, val_contexts)

        val_encodings = tokenizer(val_contexts, val_questions,
                                  truncation=True, padding=True)

        self._add_token_positions(val_encodings, val_answers, tokenizer)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            {key: val_encodings[key]
                for key in ['input_ids', 'attention_mask']},
            {key: val_encodings[key]
                for key in ['start_positions', 'end_positions']}
        ))

        # Keras will expect a tuple when dealing with labels
        val_dataset = val_dataset.map(lambda x, y: (
            x, (y['start_positions'], y['end_positions'])))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # fit model
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.evaluate(val_dataset.shuffle(
            1000).batch(batch_size), batch_size=batch_size)

    @staticmethod
    def _compute_softmax(scores):
        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs

    @staticmethod
    def _add_token_positions(encodings, answers, tokenizer):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(
                i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(
                i, answers[i]['answer_end'] - 1))
            # if None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length
        encodings.update({'start_positions': start_positions,
                          'end_positions': end_positions})

    @staticmethod
    def _add_end_idx(answers, contexts):
        for answer, context in zip(answers, contexts):
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)

            # sometimes squad answers are off by a character or two – fix this
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx-1:end_idx-1] == gold_text:
                answer['answer_start'] = start_idx - 1
                # When the gold label is off by one character
                answer['answer_end'] = end_idx - 1
            elif context[start_idx-2:end_idx-2] == gold_text:
                answer['answer_start'] = start_idx - 2
                # When the gold label is off by two characters
                answer['answer_end'] = end_idx - 2
