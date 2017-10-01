from sklearn.preprocessing import Normalizer


class NormalizerToggle(Normalizer):
    def __init__(self, use_normalize=True, norm='l2', copy=True):
        self.norm = norm
        self.copy = copy
        self.use_normalize = use_normalize

    def fit(self, X, y=None):
        return self

    def transform(self, X, copy=None):
        if self.use_normalize:
            return super().transform(X, copy=copy)
        else:
            return X
