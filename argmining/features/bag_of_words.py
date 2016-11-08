from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import logging
import numpy as np


def build(ngram=1, feature_name='bag_of_words'):
    pipeline = Pipeline([('transformer',
                          BagOfWords(ngram=ngram)),
                         ])
    return (feature_name, pipeline)


def tokenizer_THF_words(thf_sentence):
    return list(map(lambda token: token.text, thf_sentence.tokens))


class BagOfWords(BaseEstimator):
    def __init__(self, ngram=1, token_form='text'):
        self.ngram = ngram
        self.token_form = token_form
        self.logger = logging.getLogger()

    def fit(self, X, y):
        if self.token_form == 'text':
            self.vectorizer = CountVectorizer(tokenizer=tokenizer_THF_words,
                                              ngram_range=(self.ngram, self.ngram),
                                              lowercase=False)  # the lowercase is workaround for passing a custom class
            self.vectorizer.fit(X)
            self.logger.info("Created a vocabulary with length {}".format(len(self.vectorizer.get_feature_names())))
            return self
        else:
            raise ValueError("token_form not implemented")

    def transform(self, X):
        self.logger.debug("transform called")
        transformed = list(map(lambda x: self.transform_sentence(x), X))
        transformed = np.concatenate(transformed, axis=0)
        self.logger.debug("transform returning")
        return transformed

    def transform_sentence(self, thf_sentence):
        vectorized = self.vectorizer.transform([thf_sentence]).toarray()
        return vectorized
