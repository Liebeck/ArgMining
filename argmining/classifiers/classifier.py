from argmining.classifiers import svm
from argmining.classifiers import knn


def get_classifier(name):
    if name == 'svm':
        return svm.build(), svm.param_grid()
    elif name == 'knn':
        return knn.build(), knn.param_grid()
    else:
        raise ValueError("Unknown classifier")