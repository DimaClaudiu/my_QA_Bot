from abc import ABC, abstractmethod


class Reader(ABC):

    @abstractmethod
    def predict(self, question, context, best_n=3):
        pass
