from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def param_grid(cross_validation=False):
    if cross_validation:
        return {'classifier__C': [1e3],
                'classifier__gamma': [0.005, 0.01]}
    else:
        return {'classifier__C': [1e3, 5e3, 1e4, 5e4, 1e5],
                'classifier__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}

def param_grid_linear(cross_validation=False):
    if cross_validation:
        return {'classifier__C': [1e3]}
    else:
        return {'classifier__C': [1e3, 5e3, 1e4, 5e4, 1e5]}


def build(kernel='rbf', random_state=0, C=1.0, gamma='auto'):
    return SVC(kernel=kernel, class_weight='balanced', random_state=random_state, C=C, gamma=gamma)
