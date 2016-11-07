from sklearn.pipeline import Pipeline, FeatureUnion


def pipeline(strategy, classifier):
    print(type(strategy))
    return Pipeline(steps=[('union', FeatureUnion(strategy)),
                           classifier])
