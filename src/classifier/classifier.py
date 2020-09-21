from abc import ABC, abstractmethod


class Classifier(ABC):

    @abstractmethod
    def predict(self, question, label_mappings):
        pass
