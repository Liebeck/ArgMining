from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging


def build(length=6):
    pipeline = Pipeline([('transformer',
                          LongWordCount(length=length)),
                         ])
    return ('long_word_count', pipeline)


def count_long_words(thf_sentence, length):
    count = sum(1 for x in thf_sentence.tokens if len(x.text) >= length)
    return [count]


class LongWordCount(BaseEstimator):
    def __init__(self, length=6):
        self.logger = logging.getLogger()
        self.length = length

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = list(map(lambda x: count_long_words(x, self.length), X))
        return transformed
