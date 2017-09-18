from argmining.classifiers import svm
from argmining.classifiers import knn
from argmining.classifiers import random_forest


def get_classifier(name, cross_validation=False):
    if name == 'svm':
        return svm.build(), svm.param_grid(cross_validation=cross_validation)
    if name == 'svm-linear':
        return svm.build(kernel='linear'), svm.param_grid_linear(cross_validation=cross_validation)
    elif name == 'knn':
        return knn.build(), knn.param_grid(cross_validation=cross_validation)
    elif name == 'rf':
        return random_forest.build(), random_forest.param_grid(cross_validation=cross_validation)
    else:
        raise ValueError("Unknown classifier")


def create_classifier(name, classifier_params):
    if name == 'svm':
        return svm.build(**classifier_params)
    elif name == 'svm-linear':
        classifier_params['kernel'] = 'linear'
        print(classifier_params)
        return svm.build(**classifier_params)
    elif name == 'knn':
        return knn.build(**classifier_params)
    elif name == 'rf':
        return random_forest.build(**classifier_params)
    else:
        raise ValueError("Unknown classifier")
