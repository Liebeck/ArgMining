from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging
import numpy as np


def build():
    pipeline = Pipeline([('transformer',
                          SentiWSAveragePolarity()),
                         ])
    return ('polarity_sentiws_average', pipeline)


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


class SentiWSAveragePolarity(BaseEstimator):
    def __init__(self):
        self.logger = logging.getLogger()

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = list(map(lambda x: extract_average_polarity(x), X))
        return transformed
