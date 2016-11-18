from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


class SelectKBestToggle(SelectKBest):
    def __init__(self, k=30, score_func=f_classif):
        print('k: {}'.format(k))
        self.k = k
        self.score_func=score_func
        # super(SelectKBest, self).__init__(score_func)


grid = {
    #'feature_selection__k': [5, 10, 20],
    'feature_selection': [SelectKBestToggle(score_func=chi2, k=5),
                          SelectKBestToggle(score_func=chi2, k=10)],
    'classifier__gamma': [0.005, 0.01]
}

digits = load_digits()

pipeline = Pipeline([('feature_selection', SelectKBestToggle(score_func=chi2, k=1)),
                     ('classifier', SVC())])
grid = GridSearchCV(pipeline, cv=2, n_jobs=1, param_grid=grid, verbose=2)
grid.fit(digits.data, digits.target)
