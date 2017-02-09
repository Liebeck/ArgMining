from sklearn.ensemble import RandomForestClassifier


def param_grid(cross_validation=False):
    if cross_validation:
        return {"classifier__n_estimators": [10, 20, 30, 40],
                "classifier__max_depth": [5],
                "classifier__max_features": [1, 3, 10]}
    else:
        return {"classifier__n_estimators": [10, 20, 30, 40],
                "classifier__max_depth": [5, 10, 20, 30, 50, 100, None],
                "classifier__max_features": [1, 3, 10],
                "classifier__min_samples_split": [2, 3, 10],
                "classifier__min_samples_leaf": [1, 3, 10],
                "classifier__bootstrap": [True, False],
                "classifier__criterion": ["gini", "entropy"]}


def build(random_state=0, n_jobs=1):
    return RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=n_jobs)
