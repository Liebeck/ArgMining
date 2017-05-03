from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging
import numpy as np


def build(embedding_length=100, stopwords=None):
    pipeline = Pipeline([('transformer',
                          EmbeddingCentroid(embedding_length, stopwords=stopwords)),
                         ])
    return ('embedding_centroid', pipeline)


def transform_sentence(thf_sentence, embedding_length, stopwords=None):
    values = []
    logger = logging.getLogger()
    for token in thf_sentence.tokens:
        if stopwords is not None:
            key = token.get_key('lowercase')
            if key not in stopwords:
                if token.embedding is not None:
                    values.append(token.embedding)
        else:
            if token.embedding is not None:
                values.append(token.embedding)
    if not values:
        val = np.zeros(embedding_length)
        # logger.debug(val)
        return val
    else:
        arr = np.array(values)
        # logger.debug(arr)
        val = np.mean(arr, axis=0)
        # logger.debug(val)
        return val


class EmbeddingCentroid(BaseEstimator):
    def __init__(self, embedding_length, stopwords=None):
        self.logger = logging.getLogger()
        self.embedding_length = embedding_length
        self.stopwords = stopwords

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = list(map(lambda x: transform_sentence(x, self.embedding_length, self.stopwords), X))
        return transformed
