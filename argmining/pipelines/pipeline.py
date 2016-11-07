from sklearn.pipeline import Pipeline, FeatureUnion


def pipeline(strategy, classifier):
    return Pipeline(steps=[('union', FeatureUnion(strategy)),
                           classifier])
