from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import logging
import numpy as np
from argmining.transformers.normalizer_toggle import NormalizerToggle
from spacy.orth import word_shape


def build(feature_name='bag_of_words', ngram_range=(1, 1), token_form='text', lowercase=False, normalize=False,
          min_df=1, max_features=None, stopwords=None):
    pipeline = Pipeline(
        [('transformer',
          BagOfWords(ngram_range=ngram_range, token_form=token_form, lowercase=lowercase,
                     min_df=min_df,
                     max_features=max_features, stopwords=stopwords)),
         ('normalizer', NormalizerToggle(use_normalize=normalize))
         ])
    return (feature_name, pipeline)


def tokenizer_THF_words(thf_sentence):
    return list(map(lambda token: token.text, thf_sentence.tokens))


def tokenizer_THF_words_lowercase(thf_sentence):
    return list(map(lambda token: token.text.lower(), thf_sentence.tokens))


def tokenizer_shape(thf_sentence):
    return list(map(lambda token: token.spacy_shape, thf_sentence.tokens))


def tokenizer_shape_lemma(thf_sentence):
    words = []
    for token in thf_sentence.tokens:
        if token.iwnlp_lemma is not None and len(token.iwnlp_lemma) == 1:
            words.append(word_shape(token.iwnlp_lemma[0]))
        else:
            words.append(token.spacy_shape)
    return words


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
    def __init__(self, ngram_range=(1, 1), token_form='text', lowercase=False, min_df=1,
                 max_features=None, stopwords=None):
        self.ngram_range = ngram_range
        self.token_form = token_form
        self.logger = logging.getLogger()
        self.lowercase = lowercase
        self.min_df = min_df
        self.max_features = max_features
        self.stopwords = stopwords

    def get_tokenizer(self):
        if self.token_form == 'text' and not self.lowercase:
            return tokenizer_THF_words
        elif self.token_form == 'text' and self.lowercase:
            return tokenizer_THF_words_lowercase
        elif self.token_form == 'IWNLP_lemma' and not self.lowercase:
            return tokenizer_THF_lemma
        elif self.token_form == 'IWNLP_lemma' and self.lowercase:
            return tokenizer_THF_lemma_lowercase
        elif self.token_form == 'shape':
            return tokenizer_shape
        elif self.token_form == 'shape_lemma':
            return tokenizer_shape_lemma

    def fit(self, X, y):
        tokenizer = self.get_tokenizer()
        self.vectorizer = CountVectorizer(tokenizer=tokenizer,
                                          ngram_range=self.ngram_range,
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
