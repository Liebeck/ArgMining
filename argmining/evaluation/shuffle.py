from sklearn.utils import shuffle
import logging

def shuffle_training_Set(X_train, y_train, random_state):
    logger = logging.getLogger()
    logger.info("Shuffling with random_state: {}".format(random_state))
    if random_state is None:
        return X_train, y_train
    else:
        X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
        return X_train, y_train