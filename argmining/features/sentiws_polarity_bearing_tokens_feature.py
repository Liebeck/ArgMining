from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging


def build():
    pipeline = Pipeline([('transformer',
                          SentiWSPolarityBearingTokens()),
                         ])
    return ('polarity_sentiws_polarity_bearing_tokens', pipeline)


def count_polarity_bearing_tokens(thf_sentence):
    token_count = 0
    for token in thf_sentence.tokens:
        if token.polarity is not None:
            token_count += 1
    return [token_count]


class SentiWSPolarityBearingTokens(BaseEstimator):
    def __init__(self, feature):
        self.logger = logging.getLogger()

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = list(map(lambda x: self.extract_average_polarity(x), X))
        return transformed
