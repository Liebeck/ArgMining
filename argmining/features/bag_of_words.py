from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


def build(ngram=1, feature_name='bag_of_words'):
    pipeline = Pipeline([('transformer',
                          BagOfWords(ngram=ngram)),
                         ])
    return (feature_name, pipeline)


class BagOfWords(BaseEstimator):
    def __init__(self, ngram=1):
        self.ngram = ngram

    def fit(self, X, y):
        return self

    def transform(self, X):
        return list(map(lambda x: self.transform_sentence(x), X))

    def transform_sentence(self, x):
        return [len(x.tokens)] # todo: replace with one hot encoding
