from sklearn.utils import shuffle
import logging


def shuffle_training_Set(X_train, y_train, random_state):
    logger = logging.getLogger()
    if random_state is None:
        logger.info("Using the original order of the dataset")
        return X_train, y_train
    else:
        logger.info("Shuffling with random_state: {}".format(random_state))
        X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
        return X_train, y_train
