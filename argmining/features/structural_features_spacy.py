from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging


def build(use_sentence_length=True):
    pipeline = Pipeline([('transformer',
                          StructuralFeaturesSpacy(use_sentence_length=use_sentence_length)),
                         ])
    return ('structural_features_spacy', pipeline)


def transform_sentence(thf_sentence, use_sentence_length=True):
    values = []
    sentence_number = int(thf_sentence.uniqueID[thf_sentence.uniqueID.index('s') + 1:])
    values.append(sentence_number)
    words = list(map(lambda token: token.text, thf_sentence.tokens))
    comma_relative = 0
    dot_relative = 0
    for token in thf_sentence.tokens:
        if token.text == ',' or token.text == b',':
            comma_relative += 1
        if token.text == '.' or token.text == b'.':
            dot_relative += 1
    comma_relative = comma_relative / float(len(words))
    dot_relative = dot_relative / float(len(words))
    values.append(comma_relative)
    values.append(dot_relative)
    if use_sentence_length:
        values.append(len(thf_sentence.tokens))
    link_count = 0
    punctuation_count = 0
    for token in thf_sentence.tokens:
        if token.spacy_like_url:
            link_count += 1
        if token.spacy_is_punct:
            punctuation_count += 1
    values.append(float(link_count))
    values.append(float(punctuation_count))
    last_word = words[len(words) - 1]
    if last_word == '.' or last_word == b'.':
        values.extend([1.0, 0.0, 0.0, 0.0])
    elif last_word == '!' or last_word == b'!':
        values.extend([0.0, 1.0, 0.0, 0.0])
    elif last_word == '?' or last_word == b'?':
        values.extend([0.0, 0.0, 1.0, 0.0])
    else:
        values.extend([0.0, 0.0, 0.0, 1.0])
    return values


class StructuralFeaturesSpacy(BaseEstimator):
    def __init__(self, use_sentence_length=True):
        self.logger = logging.getLogger()
        self.use_sentence_length = use_sentence_length

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = list(map(lambda x: self.transform_sentence(x), X))
        return transformed
