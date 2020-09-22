import math

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

from reader.reader import Reader


class TfReader(Reader):

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = TFAutoModelForQuestionAnswering.from_pretrained(
            model_path)

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
                       probabilities[0][answer_end-1])/2

        return answer, probability

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
