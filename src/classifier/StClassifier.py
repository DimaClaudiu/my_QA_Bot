from simpletransformers.classification import ClassificationModel
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from classifier.classifier import Classifier


class StClassifier(Classifier):
    def __init__(self, model_path, num_labels):
        self.model = ClassificationModel(
            'roberta', model_path, num_labels=num_labels)

    def predict(self, question, label_mappings):
        inpt = self._clean_text(question)
        predictions, raw_outputs = self.model.predict([inpt])

        return label_mappings[predictions[0]]

    @staticmethod
    def _clean_text(text, max_len=128):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        words = [word for word in word_tokens if word.isalpha()]

        filtered_words = [w for w in words if not w in stop_words]

        return ' '.join(filtered_words)[0:max_len-1].lower()
