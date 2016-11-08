from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def param_grid():
    return {'classifier__C': [1e3],
            'classifier__gamma': [0.005, 0.01]}


def build(kernel='rbf', random_state=0):
    return SVC(kernel=kernel, class_weight='balanced', random_state=random_state)
