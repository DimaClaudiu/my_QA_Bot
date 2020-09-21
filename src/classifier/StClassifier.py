from simpletransformers.classification import ClassificationModel
from classifier.classifier import Classifier
from utils.transformer_utils import clean_text


class StClassifier(Classifier):
    def __init__(self, model_path, num_labels):
        self.model = ClassificationModel(
            'roberta', model_path, num_labels=num_labels)

    def predict(self, question, label_mappings):
        inpt = clean_text(question)
        predictions, raw_outputs = self.model.predict([inpt])

        return label_mappings[predictions[0]]
