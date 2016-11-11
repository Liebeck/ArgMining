from argmining.classifiers import svm
from argmining.classifiers import knn
from argmining.classifiers import random_forest


def get_classifier(name):
    if name == 'svm':
        return svm.build(), svm.param_grid()
    elif name == 'knn':
        return knn.build(), knn.param_grid()
    elif name == 'rf':
        return random_forest.build(), random_forest.param_grid()
    else:
        raise ValueError("Unknown classifier")
