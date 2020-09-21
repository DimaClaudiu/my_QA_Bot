from transformers import AutoTokenizer, AutoModelForQuestionAnswering, RobertaForQuestionAnswering, RobertaTokenizerFast
import torch
from reader.reader import Reader
import math


class PtReader(Reader):

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    def predict(self, question, context, best_n=3):

        inputs = self.tokenizer(
            question, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        answer_start_scores, answer_end_scores = self.model(**inputs)

        answer_start = torch.argmax(
            answer_start_scores
        )
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        probabilities = []

        for start_score in answer_start_scores.cpu().detach().numpy():
            probabilities.append(self._compute_softmax(start_score))

        return answer, probabilities

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
