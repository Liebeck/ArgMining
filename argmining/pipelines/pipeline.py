from sklearn.pipeline import Pipeline, FeatureUnion


def pipeline(transformers, classifier):
    return Pipeline(steps=[('union', FeatureUnion(transformers)),
                           classifier])
