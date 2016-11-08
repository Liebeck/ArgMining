from argmining.classifiers import svm


def get_classifier(name):
    if name == 'svm':
        return svm.build(), svm.param_grid()
    else:
        raise ValueError("Unknown classifier")