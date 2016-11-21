from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging
import numpy as np

def build():
    pipeline = Pipeline([('transformer',
                          PolaritySentiWS()),
                         ])
    return ('polarity_sentiws', pipeline)


def extract_average_polarity(thf_sentence):
    polarity_scores = []
    for token in thf_sentence.tokens:
        if token.polarity is not None:
            polarity_scores.append(token.polarity)
    print(polarity_scores)
    print(np.mean(polarity_scores))
    if not polarity_scores:
        return [0.0]
    else:
        return [np.mean(polarity_scores)]


class PolaritySentiWS(BaseEstimator):
    def __init__(self):
        self.logger = logging.getLogger()

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = list(map(lambda x: self.transform_sentence(x), X))
        return transformed