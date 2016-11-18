from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import logging
import numpy as np
from argmining.transformers.normalizer_toggle import NormalizerToggle


def build(feature_name='bag_of_words'):
    pipeline = Pipeline([('transformer', BagOfWords()),
                         ('normalizer', NormalizerToggle())
                         ])
    return (feature_name, pipeline)


def tokenizer_THF_words(thf_sentence):
    return list(map(lambda token: token.text, thf_sentence.tokens))


def tokenizer_THF_words_lowercase(thf_sentence):
    return list(map(lambda token: token.text.lower(), thf_sentence.tokens))


def tokenizer_THF_lemma(thf_sentence):
    words = []
    for token in thf_sentence.tokens:
        if token.iwnlp_lemma is not None and len(token.iwnlp_lemma) == 1:
            words.append(token.iwnlp_lemma[0])
        else:
            words.append(token.text)
    return words


def tokenizer_THF_lemma_lowercase(thf_sentence):
    words = []
    for token in thf_sentence.tokens:
        if token.iwnlp_lemma is not None and len(token.iwnlp_lemma) == 1:
            words.append(token.iwnlp_lemma[0].lower())
        else:
            words.append(token.text.lower())
    return words


class BagOfWords(BaseEstimator):
    def __init__(self, ngram=1, token_form='text', lowercase=False):
        self.ngram = ngram
        self.token_form = token_form
        self.logger = logging.getLogger()
        self.lowercase = lowercase

    def get_tokenizer(self):
        if self.token_form == 'text' and not self.lowercase:
            return tokenizer_THF_words
        elif self.token_form == 'text' and self.lowercase:
            return tokenizer_THF_words_lowercase
        elif self.token_form == 'IWNLP_lemma' and not self.lowercase:
            return tokenizer_THF_lemma
        elif self.token_form == 'IWNLP_lemma' and self.lowercase:
            return tokenizer_THF_lemma_lowercase

    def fit(self, X, y):
        tokenizer = self.get_tokenizer()
        self.vectorizer = CountVectorizer(tokenizer=tokenizer,
                                          ngram_range=(self.ngram, self.ngram),
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
