from sklearn.ensemble import RandomForestClassifier


def param_grid(cross_validation=False):
    if cross_validation:
        return {"classifier__n_estimators": [10, 20, 30, 40, 100],
                "classifier__max_depth": [5],
                "classifier__max_features": [1, 3, 10]}
    else:
        return {"classifier__n_estimators": [10, 20, 30, 40, 100],
                "classifier__max_depth": [5, 25, 50, 100, None],
                "classifier__max_features": [1, 0.2, 0.5, "auto", "log2"],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 5],
                "classifier__bootstrap": [True],
                "classifier__criterion": ["gini"]}


def build(random_state=0, n_jobs=1, min_samples_split=2, n_estimators=10, max_depth=None, max_features='auto',
          min_samples_leaf=1, bootstrap=True, criterion='gini'):
    return RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=n_jobs,
                                  min_samples_split=min_samples_split, n_estimators=n_estimators, max_depth=max_depth,
                                  max_features=max_features, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap,
                                  criterion=criterion)
