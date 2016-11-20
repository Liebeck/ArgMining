from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging


def build(use_sentence_length=True):
    pipeline = Pipeline([('transformer',
                          StructuralFeatures(use_sentence_length=use_sentence_length)),
                         ])
    return ('structural_features', pipeline)


def transform_sentence(thf_sentence, use_sentence_length=True):
    values = []
    words = list(map(lambda token: token.text, thf_sentence.tokens))
    comma_relative = words.count(',') / float(len(words))
    dot_relative = words.count('.') / float(len(words))
    values.append(comma_relative)
    values.append(dot_relative)
    if use_sentence_length:
        values.append(len(words))
    link_count = 0
    for word in words:
        if word.startswith('www.') or word.startswith('http'):
            link_count += 1
    values.append(float(link_count))
    last_word = words[len(words) - 1]

    if last_word == '.':
        values.extend([1.0, 0.0, 0.0, 0.0])
    elif last_word == '!':
        values.extend([0.0, 1.0, 0.0, 0.0])
    elif last_word == '?':
        values.extend([0.0, 0.0, 1.0, 0.0])
    else:
        values.extend([0.0, 0.0, 0.0, 1.0])
    return values


class StructuralFeatures(BaseEstimator):
    def __init__(self, use_sentence_length=True):
        self.logger = logging.getLogger()
        self.use_sentence_length = use_sentence_length

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = list(map(lambda x: self.transform_sentence(x), X))
        return transformed
