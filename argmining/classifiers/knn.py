from sklearn.neighbors import KNeighborsClassifier


def param_grid():
    return {"classifier__n_neighbors": [6, 7, 8, 9, 10, 11, 12, 13],
            "classifier__leaf_size": [20, 30, 40],
            "classifier__weights": ["distance", "uniform"]
            }


def build():
    return KNeighborsClassifier()
