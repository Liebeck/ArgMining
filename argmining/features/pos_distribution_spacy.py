from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging
from sklearn.preprocessing import Normalizer
from argmining.models.pos_universal_google import POS_UNIVERSAL_GOOGLE
from argmining.models.stts import STTS_TAGSET_SPACY
from collections import OrderedDict
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


def build(coarse_grained=True):
    pipeline = Pipeline([('transformer', POSDistributionSpacy(coarse_grained=coarse_grained)),
                         ('normalizer', Normalizer())
                         ])
    return ('pos_distribution_spacy', pipeline)


def build_feature_selection(coarse_grained=True, k=5):
    pipeline = Pipeline([('transformer', POSDistributionSpacy(coarse_grained=coarse_grained)),
                         ('feature_selection', SelectKBest(chi2, k=k)),
                         ('normalizer', Normalizer())
                         ])
    return ('pos_distribution_spacy', pipeline)


def get_pos_histogram(pos_list, tag_set):
    histogram = OrderedDict.fromkeys(tag_set, 0)
    for entry in pos_list:
        histogram[entry] += 1
    values = []
    for key, value in histogram.items():
        values.append(value)
    histogram = np.array(values, dtype=np.float64)
    return histogram


class POSDistributionSpacy(BaseEstimator):
    def __init__(self, coarse_grained=True):
        self.logger = logging.getLogger()
        self.coarse_grained = coarse_grained

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = list(map(lambda x: self.transform_sentence(x), X))
        return transformed

    def transform_sentence(self, thf_sentence):
        if self.coarse_grained:
            pos_list = list(map(lambda x: x.spacy_pos_stts, thf_sentence.tokens))
            distribution = get_pos_histogram(pos_list, STTS_TAGSET_SPACY)
        else:
            pos_list = list(map(lambda x: x.spacy_pos_universal_google, thf_sentence.tokens))
            distribution = get_pos_histogram(pos_list, POS_UNIVERSAL_GOOGLE)
        return distribution
