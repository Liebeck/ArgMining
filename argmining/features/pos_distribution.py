from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging
from sklearn.preprocessing import Normalizer
from argmining.models.stts import STTS_TAGSET
from collections import OrderedDict
import numpy as np


def build(use_STTS=True):
    if use_STTS:
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


def get_STTS_histogram(pos_list):
    histogram = OrderedDict.fromkeys(STTS_TAGSET, 0)
    for entry in pos_list:
        histogram[entry] += 1
    histogram = np.array(histogram.values(), dtype=np.float64)

    list(d.items())

    print(histogram)
    return histogram


class POSDistribution(BaseEstimator):
    def __init__(self, use_STTS):
        self.logger = logging.getLogger()
        self.use_STTS = use_STTS

    def fit(self, X, y):
        return self

    def transform(self, X):
        self.logger.debug("transform called")
        transformed = list(map(lambda x: self.transform_sentence(x), X))
        # transformed = np.concatenate(transformed, axis=0)
        self.logger.debug("transform returning")
        return None

    def transform_sentence(self, thf_sentence):
        pos_list = list(map(lambda x: x.pos_tag, thf_sentence.tokens))
        if self.use_STTS:
            distribution = get_STTS_histogram(pos_list)
            return distribution
        else:
            return None
