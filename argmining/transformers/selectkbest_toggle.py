from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


class SelectKBestToggle(SelectKBest):
    def __init__(self, score_func=f_classif, k=10, use_feature_selection=False):
        print(k)
        print(use_feature_selection)
        self.use_feature_selection = use_feature_selection
        super(SelectKBest, self).__init__(score_func)
        # self.use_feature_selection = use_feature_selection
        self.k = k
        if not self.use_feature_selection:
            print("not using feature selection")
            self.k = 'all'


            # def transform(self, X):
            #     if self.use_feature_selection:
            #         # return super(SelectKBestToggle).transform(X)
            #         return super.transform(X)
            #     else:
            #         return X

            # def fit(self, X, y=None):
            #     if self.use_feature_selection:
            #         print('using feature selection')
            #         # return super(_BaseFilter, self).fit(X, y)
            #         return super(SelectKBestToggle, self).fit(X, y)
            #         # return super.fit(X, y)
            #     else:
            #         print('not using feature selection')
            #         # super(SelectKBestToggle, self).fit(X, y)
            #         # check_is_fitted(self, 'scores_')
            #         return self
            #
            #     def _get_support_mask(self):
            #         if self.use_feature_selection:
            #
            #         else:
            #
            #
            #         check_is_fitted(self, 'scores_')
            #         if self.k == 'all':
            #             return np.ones(self.scores_.shape, dtype=bool)
            #         elif self.k == 0:
            #             return np.zeros(self.scores_.shape, dtype=bool)
            #         else:
            #             scores = _clean_nans(self.scores_)
            #             mask = np.zeros(scores.shape, dtype=bool)
            #
            #             # Request a stable sort. Mergesort takes more memory (~40MB per
            #             # megafeature on x86-64).
            #             mask[np.argsort(scores, kind="mergesort")[-self.k:]] = 1
            #             return mask
