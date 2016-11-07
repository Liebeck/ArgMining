from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


def build(ngram='1'):
    pipeline = Pipeline([('transformer',
                          BagOfWords(ngram=ngram)),
                         ])
    return ('bag_of_words', pipeline)


class BagOfWords(BaseEstimator):
    def __init__(self, ngram=1):
        self.ngram = ngram

    def fit(self, X, y):
        return self

    def transform(self, raw_submissions):
        return map(lambda x: self._transform(x), raw_submissions)
