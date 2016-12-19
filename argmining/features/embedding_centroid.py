from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging
import numpy as np

def build(embedding_length=100):
    pipeline = Pipeline([('transformer',
                          EmbeddingCentroid(embedding_length)),
                         ])
    return ('embedding_centroid', pipeline)


def transform_sentence(thf_sentence, embedding_length):
    values = []
    logger = logging.getLogger()
    for token in thf_sentence.tokens:
        if token.embedding is not None:
            values.append(token.embedding)
    if not values:
        val = [np.zeros(embedding_length)]
        logger.debug(val)
        return val
    else:
        val = [np.mean(values)]
        logger.debug(val)
        return val




class EmbeddingCentroid(BaseEstimator):
    def __init__(self, embedding_length):
        self.logger = logging.getLogger()
        self.embedding_length = embedding_length

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = list(map(lambda x: transform_sentence(x, self.embedding_length), X))
        return transformed
