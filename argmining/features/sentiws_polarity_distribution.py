from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging
import numpy as np


def build(bins=10):
    pipeline = Pipeline([('transformer',
                          SentiWSPolarityDistribution(bins=bins)),
                         ])
    return ('polarity_sentiws_distribution', pipeline)


def extract_polarity_tokens(thf_sentence):
    polarity_tokens = []
    for token in thf_sentence.tokens:
        if token.polarity is not None:
            polarity_tokens.append(token.polarity)
    if not polarity_tokens:
        return [0]
    else:
        return polarity_tokens


class SentiWSPolarityDistribution(BaseEstimator):
    def __init__(self, bins=10, density=None):
        self.bins = bins
        self.density = density
        self.logger = logging.getLogger()

    def fit(self, X, y):
        all_polarity_values = []
        for thf_sentence in X:
            all_polarity_values.extend(extract_polarity_tokens(thf_sentence))
        histogram, edges = np.histogram(all_polarity_values, bins=self.bins, density=self.density)
        self.edges = edges
        return self

    def transform(self, X):
        transformed = list(map(lambda x: self.transform_sentence(x), X))
        return transformed

    def transform_sentence(self, thf_sentence):
        polarity_tokens = extract_polarity_tokens(thf_sentence)
        histogram, edges = np.histogram(polarity_tokens, bins=self.edges, density=False)
        return histogram
