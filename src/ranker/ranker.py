from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class Ranker(ABC):

    @abstractmethod
    def get_best_contexts(self, question, contexts, top_k):
        pass
