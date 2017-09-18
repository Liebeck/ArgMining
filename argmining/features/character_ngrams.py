from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import logging
import numpy as np
from argmining.transformers.normalizer_toggle import NormalizerToggle


def build(feature_name='character_ngram', min_n=1, max_n=1, normalize=False,
          min_df=1, max_features=None, stopwords=None):
    pipeline = Pipeline(
        [('transformer',
          CharacterNGrams(min_n=min_n, max_n=max_n, min_df=min_df,
                          max_features=max_features, stopwords=stopwords)),
         ('normalizer', NormalizerToggle(use_normalize=normalize))
         ])
    return (feature_name, pipeline)


def tokenizer_THF_words(thf_sentence):
    return list(map(lambda token: token.text, thf_sentence.tokens))


def get_ngrams_sentence(thf_sentence, min_n, max_n):
    tokens = tokenizer_THF_words(thf_sentence)
    return get_ngrams(tokens, min_n, max_n)


def get_ngrams(tokens, min_n, max_n):
    # based on 'char_wb' from CountVectorizer
    n_grams = []
    for w in tokens:
        w = ' ' + w
        w_len = len(w)
        for n in range(min_n, max_n + 1):
            offset = 0
            while offset + n < w_len:
                offset += 1
                n_grams.append(w[offset:offset + n])
            if offset == 0:  # count a short word (w_len < n) only once
                break
    return n_grams


class CharacterNGrams(BaseEstimator):
    def __init__(self, min_n=1, max_n=1, min_df=1,
                 max_features=None, stopwords=None):
        self.min_n = min_n
        self.max_n = max_n
        self.logger = logging.getLogger()
        self.min_df = min_df
        self.max_features = max_features
        self.stopwords = stopwords

    def fit(self, X, y):
        self.vectorizer = CountVectorizer(tokenizer=lambda text: get_ngrams_sentence(text, self.min_n, self.max_n),
                                          ngram_range=(1, 1),
                                          min_df=self.min_df,
                                          max_features=self.max_features,
                                          stop_words=self.stopwords,
                                          lowercase=False)  # the lowercase is workaround for passing a custom class
        self.vectorizer.fit(X)
        self.logger.info("Created a vocabulary with length {}".format(len(self.vectorizer.get_feature_names())))
        return self

    def transform(self, X):
        transformed = list(map(lambda x: self.transform_sentence(x), X))
        transformed = np.concatenate(transformed, axis=0)
        return transformed

    def transform_sentence(self, thf_sentence):
        vectorized = self.vectorizer.transform([thf_sentence]).toarray()
        vectorized = vectorized.astype(np.float64)
        return vectorized
