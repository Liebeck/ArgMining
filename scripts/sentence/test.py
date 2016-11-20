from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin


class SelectKBestToggle(SelectKBest):
    def __init__(self, k=30, score_func=f_classif, use_feature_selection=False):
        # print('k: {}'.format(k))
        self.k = k
        self.use_feature_selection = use_feature_selection
        self.score_func = score_func
        # super(SelectKBest, self).__init__(score_func)

    def set_params(self, **params):
        if 'use_feature_selection' in params:
            if not params['use_feature_selection']:
                print("FS is disabled")
                BaseEstimator.set_params(self, k='all')
                # params['k'] = 'all'
                # print(params['use_feature_selection'])
        # print(params['k'])
        print(params)

        # super(BaseEstimator, self).set_params(**params)
        BaseEstimator.set_params(self, **params)

        # def transform(self, X):
        # print(self)

        # super(SelectorMixin, self).transform(X)
        # SelectorMixin.transform(self, X)


grid = [{
    'feature_selection__k': [5, 10, 20],
    'feature_selection__use_feature_selection': [True],
    'classifier__gamma': [0.005, 0.01]
}, {
    'feature_selection__use_feature_selection': [False],
    'classifier__gamma': [0.005, 0.01]
}
]

digits = load_digits()

pipeline = Pipeline([('feature_selection', SelectKBestToggle(score_func=chi2, k=1)),
                     ('classifier', SVC())])
grid = GridSearchCV(pipeline, cv=2, n_jobs=1, param_grid=grid, verbose=2)
grid.fit(digits.data, digits.target)
