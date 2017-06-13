from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging


def build():
    pipeline = Pipeline([('transformer', LDADistribution()),
                         ])
    return ('lda_distribution', pipeline)


class LDADistribution(BaseEstimator):
    def __init__(self):
        self.logger = logging.getLogger()

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = list(map(lambda x: x.lda_embedding, X))
        return transformed
