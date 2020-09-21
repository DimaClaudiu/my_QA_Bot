from simpletransformers.question_answering import QuestionAnsweringModel
from reader.reader import Reader


class StReader(Reader):

    def __init__(self, model_path):
        self.model = model = QuestionAnsweringModel(
            'roberta', model_path, args={'reprocess_input_data': True})

    def predict(self, question, context, best_n=3):
        qc = [{'context': context, 'qas': [
            {'question': question, 'id': '0'}]}]

        result = self.model.predict(qc, best_n)

        answers = result[0]
        probabilities = result[1]

        return answers, probabilities
