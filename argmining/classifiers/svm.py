from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def param_grid(cross_validation=False):
    if cross_validation:
        return {'classifier__C': [1e3],
                'classifier__gamma': [0.005, 0.01]}
    else:
        return {'classifier__C': [1e-1, 1e0, 1e1, 1e2, 1e3, 5e3, 1e4, 5e4],
                'classifier__gamma': [1e-4, 1e-5, 1e-3, 5e-3, 1e-2, 1e-1, 1e0, 1e1]}


def param_grid_linear(cross_validation=False):
    if cross_validation:
        return {'classifier__C': [1e3]}
    else:
        return {'classifier__C': [1e-3, 1e-2, 1e-1, 1e0, 1e2, 1e3, 5e3]}


def build(kernel='rbf', random_state=0, C=1.0, gamma='auto'):
    return SVC(kernel=kernel, class_weight='balanced', random_state=random_state, C=C, gamma=gamma)
