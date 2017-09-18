from sklearn.neighbors import KNeighborsClassifier


def param_grid(cross_validation=False):
    if cross_validation:
        return {"classifier__n_neighbors": [6, 7, 8, ],
                "classifier__leaf_size": [20, 30],
                "classifier__weights": ["distance", "uniform"]
                }
    else:
        return {"classifier__n_neighbors": [6, 7, 8, 9, 10, 11, 12, 13],
                "classifier__leaf_size": [20, 30, 40],
                "classifier__weights": ["distance", "uniform"]
                }


def build(n_neighbors=5, leaf_size=30, weights='uniform'):
    return KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, weights=weights)
