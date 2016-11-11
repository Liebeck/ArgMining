from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging
from sklearn.preprocessing import Normalizer
from argmining.models.stts import STTS_TAGSET
from argmining.models.uts import UTS_TAGSET, get_UTS_tag
from collections import OrderedDict
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2


def build(use_STTS=True, use_feature_selection=False, feature_selection_k=10):
    if use_STTS:
        if use_feature_selection:
            pipeline = Pipeline([('transformer',
                                  POSDistribution(use_STTS=True)),
                                 ('feature_selection', SelectKBest(chi2, k=feature_selection_k)),
                                 ('normalizer', Normalizer())
                                 ])
        else:

            pipeline = Pipeline([('transformer',
                                  POSDistribution(use_STTS=True)),
                                 ('normalizer', Normalizer())
                                 ])
    else:
        pipeline = Pipeline([('transformer',
                              POSDistribution(use_STTS=False)),
                             ('normalizer', Normalizer())
                             ])
    return ('pos_distribution', pipeline)


def get_pos_histogram(pos_list, tag_set):
    histogram = OrderedDict.fromkeys(tag_set, 0)
    for entry in pos_list:
        histogram[entry] += 1
    values = []
    for key, value in histogram.items():
        values.append(value)
    histogram = np.array(values, dtype=np.float64)
    return histogram


class POSDistribution(BaseEstimator):
    def __init__(self, use_STTS):
        self.logger = logging.getLogger()
        self.use_STTS = use_STTS

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = list(map(lambda x: self.transform_sentence(x), X))
        return transformed

    def transform_sentence(self, thf_sentence):
        if self.use_STTS:
            pos_list = list(map(lambda x: x.pos_tag, thf_sentence.tokens))
            distribution = get_pos_histogram(pos_list, STTS_TAGSET)
        else:
            pos_list = list(map(lambda x: get_UTS_tag(x.pos_tag), thf_sentence.tokens))
            distribution = get_pos_histogram(pos_list, UTS_TAGSET)
        return distribution
