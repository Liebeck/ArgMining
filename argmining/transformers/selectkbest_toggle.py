from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.base import BaseEstimator
import numpy as np


class SelectKBestToggle(SelectKBest):
    def __init__(self, score_func=f_classif, k=10, use_feature_selection=False):
        self.z_use_feature_selection = use_feature_selection
        self.k = k
        super(SelectKBest, self).__init__(score_func)

    def set_params(self, **params):
        BaseEstimator.set_params(self, **params)

        print(params)
        if 'use_feature_selection' in params:
            if not params['use_feature_selection']:
                print("FS is disabled")
                BaseEstimator.set_params(self, k='all')
        # # params['k'] = 'all'
        #         # print(params['use_feature_selection'])
        #     BaseEstimator.set_params(self, **params)
        # if 'k' in params:
        #     print(params['k'])
        #     if self.use_feature_selection:
        # # super(BaseEstimator, self).set_params(**params)
        #         BaseEstimator.set_params(self, **params)
        #     else:
        #         print("don't set k, use_feature_select is false")
        print(BaseEstimator.get_params(self))

        # def _get_support_mask(self):
        # print("_get_support_mask called")
        # print(self.use_feature_selection)
        # return super(SelectKBest, self)._get_support_mask()
        # if not self.use_feature_selection:
        # return np.ones(self.scores_.shape, dtype=bool)
        # else:
        # return super(SelectKBest, self)._get_support_mask()
